### Create dummy LoRA parameters
```python
# vllm/lora/lora.py
@classmethod
    def create_dummy_lora_weights(
            cls,
            module_name: str,
            input_dim: int,
            output_dim: int,
            rank: int,
            dtype: torch.dtype,
            device: torch.types.Device,
            embeddings_tensor_dim: Optional[int] = None,
            bias_enabled: Optional[bool] = False) -> "LoRALayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        lora_a = torch.zeros([input_dim, rank],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        lora_b = torch.zeros([rank, output_dim],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        if bias_enabled:
            bias = torch.zeros([output_dim],
                               dtype=dtype,
                               device=device, 
                               pin_memory=pin_memory)
        else:
            bias = None

        embeddings_tensor = torch.rand(
            10,
            embeddings_tensor_dim,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory) if embeddings_tensor_dim else None
        return cls(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=lora_a,
            lora_b=lora_b,
            bias=bias,
            embeddings_tensor=embeddings_tensor,
        )
```

### How to use the create_dummy_lora_weights
- when Load model parameters, call create_dummy_lora
```python
# vllm/lora/models.py
class LoRAModelManager(AdapterModelManager):
    """A manager that manages multiple LoRA-fine-tuned models."""

    def create_dummy_lora(
        self,
        lora_id: int,
        rank: int,
        scaling_factor: Optional[float],
        embedding_modules: Optional[Dict[str, str]] = None) -> LoRAModel:

        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {}, scaling_factor)
        for module_name, module in self.model.named_modules():
            bias_enabled = self.lora_config.bias_enabled
            if (not self._match_target_modules(module_name)
                    or not isinstance(module, BaseLayerWithLoRA)
                    or isinstance(module, LinearScalingRotaryEmbeddingWithLora)
                    or self._filter_unsupported_mm_module(module_name)):
                continue
            parts = module_name.split(".")
            # ... ...
            lora = LoRALayerWeights.create_dummy_lora_weights(
                                module_name,
                                module.lora_a_stacked[0].shape[-1],
                                module.lora_b_stacked[0].shape[-2],
                                rank,
                                module.lora_a_stacked[0].dtype,
                                "cpu",
                                bias_enabled=bias_enabled,
                            )
            lora.optimize()
            model.loras[module_name] = lora
        return model
```

- vllm/lora/worker_manager.py
```python
class WorkerLoRAManager(AbstractWorkerManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _manager_cls: Type[LoRAModelManager] = LoRAModelManager

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(
                lora_request.lora_int_id)
        else:
            # without keeped lora-parameters, create dummy one
            # self._adapter_manager: LoRAModelManager
            dummy_lora = self._adapter_manager.create_dummy_lora(
                lora_request.lora_int_id, rank, 1, self.embedding_modules)
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._adapter_manager.add_adapter(dummy_lora)
```

- vllm/worker/model_runner.py
```python
class GPUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TModelInputForGPU]
    _builder_cls: Type[ModelInputForGPUBuilder]


    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            # avoid create dummy lora repeatly
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):   
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/not/a/real/path",
                    )
                    # max lora numbers, with similar lora rank: LORA_WARMUP_RANK = 8
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # prepare input sequences: 
        # ... ... 

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [
            torch.tensor([], dtype=torch.float32, device=self.device)
            for _ in range(num_layers)
        ]
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        # execute LLM inference
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        # ... ...
        self.set_active_loras(model_input.lora_requests,
                              model_input.lora_mapping)
        # called: self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

## worker functions
## vllm/adapter_commons/utils.py
def set_active_adapters_worker(requests: Set[Any], mapping: Optional[Any],
                               apply_adapters_func,
                               set_adapter_mapping_func) -> None:
    apply_adapters_func(requests)
    set_adapter_mapping_func(mapping)
```

### Examples of using LoRA + llama
```python
# tests/lora/test_llama_tp.py
from vllm.lora.request import LoRARequest

def test_llama_lora(sql_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   enable_lora=True,
                   max_num_seqs=16,
                   max_loras=4,         # max number of loras
                   tensor_parallel_size=1)
    generate_and_test(llm, sql_lora_files)
```