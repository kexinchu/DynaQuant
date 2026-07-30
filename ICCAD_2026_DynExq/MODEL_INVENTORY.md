# Model Inventory and Formal-Run Mapping

This inventory separates checkpoints that can reproduce a static quantized
baseline from source checkpoints that can initialize DynaExQ. A quantized
checkpoint is not treated as an FP16/BF16 source: discarded high-precision
expert weights cannot be reconstructed from it.

| Paper model | Local checkpoint | Verified role | Remaining requirement |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | `/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound` | Static mixed-AutoRound INT4 baseline. Config/tokenizer load and a real single-GPU greedy generation passed. | Use an optimized AutoRound/GPTQ backend for formal latency runs and register the resulting artifact. |
| Qwen3-30B-A3B-Instruct-2507 | `/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned` | BF16 source for DynaExQ FP16/INT4. A real smoke run packed both tiers, released 57,982,058,496 source-expert bytes, registered all 6,144 experts on one A6000, and produced the next token `Hartford`. Current-process HBM peaked at 24,169,676,800 bytes for a 10% FP16 pool. Hugging Face revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; 27 files; 16 weight shards; 61,064,245,248 indexed tensor bytes. | Build the independent calibration ranking and run formal experiments. |
| Phi-3.5-MoE-instruct | `/home/kec23008/Models/Phi-3.5-MoE-instruct-mixed-AutoRound` | **Rejected as incomplete; not a baseline artifact.** Its index contains only layers 0--23 of the 32 layers declared by `config.json`, and omits the embedding, final norm, and language-model head. A real load therefore attempted to initialize missing tensors and exhausted the A6000. | Do not use this directory. |
| Phi-3.5-MoE-instruct | `/dev/shm/dynaexq-models/phi35-moe-bf16` | Complete BF16 source. Hugging Face revision `43688451b462a3351d8580625ebe1931adb3986d`; 35 files; 17 weight shards; 1,957 tensors; 83,746,306,688 verified tensor bytes; 32/32 declared layers. | Run DynaExQ source initialization and formal experiments from a persistent copy. |
| Phi-3.5-MoE-instruct | `/home/kec23008/Models/Phi-3.5-MoE-instruct-W4A16-AutoRound-formal` | Complete locally derived W4A16 AutoRound baseline, persisted as a byte-identical copy of the audited `/dev/shm` export. The pinned 256-by-2,048-token calibration run finished all 32 layers in 6,236.78 s with 183.64 GB peak RAM and 21.02 GB peak VRAM. Structural audit verified 11 shards, 5,285 tensors, and 22,147,709,568 tensor bytes. A real one-A6000 Triton-backend generation completed in 11.94 s of load time, peaked at 23,353,884,672 current-process HBM bytes, and generated `Hartford`. | Register formal quality/performance runs. |
| Qwen3-Next-80B-A3B-Instruct | `/dev/shm/dynaexq-models/qwen3-next-80b-bf16` | Complete BF16 source. Hugging Face revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`; 51 files; 41 weight shards; 75,944 tensors; 162,649,725,440 verified tensor bytes; 48/48 declared layers. The full-size fused 16-expert path produced 49,152 host-cache entries, released 154,618,822,656 native expert bytes, registered all 24,576 handles, passed wrapper validation and a real forward, and used 27,806,138,368 current-process HBM bytes. Initialization fell from 5,860.62 s for the legacy per-expert path to 1,063.05 s (5.51x speedup; 81.9% reduction). | Generate independent static INT4/INT2 artifacts and run formal experiments from a clean persistent snapshot. |

The full content manifests are
`results/model_manifests/qwen3_30b_a3b_instruct_2507_bf16.json`,
`results/model_manifests/phi35_moe_bf16.json`, and
`results/model_manifests/qwen3_next_80b_a3b_bf16.json`. The derived Phi
checkpoint's original export and persistent copy are separately recorded by
`results/model_manifests/phi35_moe_w4a16_autoround_formal.json` and
`results/model_manifests/phi35_moe_w4a16_autoround_formal_persistent.json`.
The persistent manifest SHA-256 is
`b0ad8c16a0e6a3314eac31157a5fce8dd1af799cc3ea25731e89444ebbde152a`;
its 22 file records and 22,152,542,913 snapshot bytes exactly match the
audited export.
Each records a SHA-256 digest for every model file. New source snapshots must
be registered with
`scripts/build_model_manifest.py`; moving revisions such as `master` are
rejected. Registration now also cross-checks every safetensors header against
the shard index, recomputes indexed tensor bytes, requires every layer declared
by the model config, and verifies embedding/final-normalization/output tensors.
This prevents an interrupted quantization export from entering the formal
artifact set. Locally quantized outputs are registered with
`scripts/build_quantized_model_manifest.py`, which binds the complete derived
checkpoint to both its source manifest and quantization provenance.

The independent quantizer corpus is
`calibration_datasets/formal/wikitext103_train_256x2048.jsonl` (SHA-256
`8988b353b026ddc168c476b2cfa905634bb776bb93eb5cafc123989da07a76b4`).
All 256 records come from the pinned WikiText-103 training split. The Phi
tokenizer measures 4,250--5,807 tokens before truncation, so every record
satisfies the requested 2,048-token AutoRound calibration length; the
quantization entry point now rejects the run before loading model weights if
even one record is shorter.

The host-packing path was also exercised at full Qwen3-30B scale. Packing
12,288 tier entries produced 72,930,557,952 host-cache bytes and released
57,982,058,496 native expert bytes. Returning dead ordinary allocation arenas
after each layer reduced measured process peak RSS to 165,740,113,920 bytes;
all 48 trim calls succeeded. This prevents temporary pin-memory copies from
accumulating across layers and is required before attempting the larger
Qwen3-Next source.

At full Qwen3-Next scale, bounded 16-expert fused packing preserves the exact
49,152 tier entries and 61,605,937,152 host-cache bytes while reducing
initialization from 5,860.62 to 1,063.05 seconds. All 48 source layers were
released and allocator trims succeeded; the resulting one-A6000 wrapper
registered 24,576 handles and completed a real forward. This is a non-formal
startup/correctness smoke check recorded in
`results/smoke_checks/qwen3_next_80b_fused_pack_startup.json`, not a manuscript
latency claim.

Formal inference remains single-GPU. On this shared two-A6000 host,
`CUDA_VISIBLE_DEVICES=1` selects one physical GPU. Other processes may retain
idle HBM on that device, but any nonzero foreign-process NVML utilization
during a measured sample invalidates it. Splitting one model across both GPUs
would change the paper's system scope and is not a valid formal run.
