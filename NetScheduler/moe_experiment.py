"""
moe_experiment.py
===================

This module provides a simple implementation of a Mixture‑of‑Experts (MoE)
layer in PyTorch that has been tailored for performance investigations on
expert parallelism.  The primary objective of the code in this file is to
expose two different deployment strategies—data parallelism (DP) and
tensor parallelism (TP)—while restricting each MoE layer to exactly one
expert and forcing the router to send every token to that single expert.

The module also contains utilities for running synthetic inference
benchmarks.  These benchmarks can be configured to generate requests of
varying input lengths at different queries per second (QPS) and will
measure two common metrics used in large language model inference:

* **Time To First Token (TTFT)** – the latency between submitting a
  request and receiving the first generated token.
* **Time Per Output Token (TPOT)** (also referred to as TPOP) – the
  average time to produce each additional token after the first.  These
  definitions follow the descriptions given in the MosaicML/Databricks
  blog on LLM inference performance【412184667459434†L221-L239】.

The code is structured to run on a single machine with multiple GPUs.  If
fewer than eight GPUs are present, the code will still run but will
replicate or split across whatever devices are available.  When only the
CPU is available, PyTorch will fall back to CPU execution; in that case,
parallelism strategies will be simulated on the CPU for functional
correctness rather than speed.

Usage
-----

To run a simple benchmark with a batch of requests all containing 128
tokens at 4 QPS using data parallelism across available GPUs:

```bash
python moe_experiment.py --mode dp --sequence-length 128 --qps 4 --duration 10
```

To run the same benchmark with tensor parallelism instead:

```bash
python moe_experiment.py --mode tp --sequence-length 128 --qps 4 --duration 10
```

The script prints aggregate TTFT and TPOT statistics at the end of the
benchmark.  See the `main` function for additional command‑line
parameters.

Limitations
-----------

The benchmark deliberately avoids performing any autograd/gradient
computations.  It uses randomly initialised models and random inputs to
isolate the cost of the forward pass.  In a real deployment you would
load pretrained weights and integrate the model into a text generation
pipeline.  Furthermore, because this environment may not have access to
multiple GPUs, the tensor parallel implementation is written to
fall back to CPU devices.  It is meant as a reference for how to
structure such a model rather than a definitive high‑performance
implementation.
"""

import argparse
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_available_devices(max_devices: int = 8) -> List[torch.device]:
    """Return a list of devices up to ``max_devices`` entries.

    If CUDA is available the returned list will be a list of CUDA devices
    (e.g. [cuda:0, cuda:1, ...]).  Otherwise the list will contain a
    single CPU device.  This helper makes the rest of the code agnostic
    to the number of GPUs present.
    """
    if torch.cuda.is_available():
        gpu_count = min(torch.cuda.device_count(), max_devices)
        return [torch.device(f"cuda:{i}") for i in range(gpu_count)]
    return [torch.device("cpu")]


class BasicExpert(nn.Module):
    """A simple feed‑forward expert comprising a single linear layer.

    The expert maps from ``input_dim`` to ``output_dim``.  In this
    benchmark we use identical input and output dimensions so that
    splitting along the output dimension for tensor parallelism is
    straightforward.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.linear(x))


class SingleExpertMoE(nn.Module):
    """Mixture‑of‑Experts layer with exactly one expert.

    In a typical MoE layer a router decides which expert(s) should
    process each token and combines the outputs accordingly.  For the
    experiments described in the user request we deliberately disable
    routing by defining only a single expert and hard coding the router
    to send all tokens to this expert.  This makes the layer behave
    identically to a standard feed‑forward network but allows us to
    exercise the same API as a full MoE layer.  Multiple layers can be
    stacked to form a larger model.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # For expert parallelism we would normally instantiate multiple
        # experts here.  Instead we create a single expert and ignore any
        # gating logic.  The gating network is retained for potential
        # future extensions but its output is disregarded.
        self.expert = BasicExpert(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)  # unused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` has shape (batch, seq_len, hidden_dim).  We reshape
        # into (batch*seq_len, hidden_dim) for processing by the
        # expert.  All tokens are sent to the single expert; there is no
        # routing to other experts.  After processing we reshape back
        # into (batch, seq_len, hidden_dim).
        b, s, h = x.shape
        flat = x.view(-1, h)
        out = self.expert(flat)
        return out.view(b, s, h)


class DataParallelMoE(nn.Module):
    """Wrap a model for data parallel inference across multiple devices.

    When using data parallelism the entire model (including the single
    expert) is replicated on each GPU.  A batch of requests is split
    across GPUs along the batch dimension.  After processing, outputs
    from each GPU are gathered on the first device and concatenated.
    """

    def __init__(self, base_model: nn.Module, devices: List[torch.device]):
        super().__init__()
        # ``torch.nn.DataParallel`` can be used when more than one GPU
        # exists.  If only one GPU or CPU is available, DataParallel
        # simply returns the original module.
        self.devices = devices
        if len(devices) > 1:
            self.model = nn.DataParallel(base_model.to(devices[0]), device_ids=list(range(len(devices))))
        else:
            # Only one device, run on CPU or single GPU without DP
            self.model = base_model.to(devices[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TensorParallelExpert(nn.Module):
    """Expert whose weight matrix is partitioned across multiple devices.

    This class illustrates how one might split a linear layer across
    several GPUs.  Each partition holds a slice of the output
    dimension.  During the forward pass each device computes its
    partial output; the outputs are then concatenated along the
    feature dimension on the first device.  For environments with no
    available GPUs the expert simply runs on CPU.
    """

    def __init__(self, input_dim: int, output_dim: int, devices: List[torch.device]):
        super().__init__()
        self.devices = devices
        self.partitions = nn.ModuleList()

        # Determine the size of each partition along the output dimension.
        # The last partition may hold a few extra elements if ``output_dim``
        # is not divisible by the number of devices.
        split_sizes = [output_dim // len(devices) for _ in devices]
        remainder = output_dim % len(devices)
        for i in range(remainder):
            split_sizes[i] += 1

        start = 0
        for size, device in zip(split_sizes, devices):
            end = start + size
            # Each sub‑module holds a slice of the output
            sub_linear = nn.Linear(input_dim, size).to(device)
            self.partitions.append(sub_linear)
            start = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is expected on the first device.  We move it to each
        # device, compute the partial output and collect results.  To
        # avoid unnecessary synchronisation we perform the computation in
        # a for loop; PyTorch handles asynchronous GPU execution.
        outputs = []
        for linear, device in zip(self.partitions, self.devices):
            # Move input to the target device
            x_device = x.to(device)
            out = F.silu(linear(x_device))
            outputs.append(out)
        # Concatenate along the feature dimension on the first device
        # Note: if running on CPU this will simply concat CPU tensors.
        return torch.cat([o.to(self.devices[0]) for o in outputs], dim=-1)


class TensorParallelMoE(nn.Module):
    """MoE layer that uses tensor parallelism for its single expert.

    In contrast to data parallelism—where the entire expert is
    replicated—tensor parallelism splits the expert's weight matrix
    across multiple devices.  The router is still disabled; all tokens
    are processed by the single expert whose output dimension is
    assembled by concatenating the partial results.  When only one
    device is available the behaviour reduces to a normal single
    expert.  See ``TensorParallelExpert`` for details of the split.
    """

    def __init__(self, hidden_dim: int, devices: List[torch.device]):
        super().__init__()
        self.expert = TensorParallelExpert(hidden_dim, hidden_dim, devices)
        self.gate = nn.Linear(hidden_dim, 1)  # unused
        self.devices = devices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` has shape (batch, seq_len, hidden_dim).  We flatten the
        # sequence dimension and process tokens by the tensor parallel
        # expert.  Finally we reshape back to (batch, seq_len, hidden_dim).
        b, s, h = x.shape
        flat = x.view(-1, h)
        out = self.expert(flat)
        return out.view(b, s, h)


@dataclass
class BenchmarkResult:
    ttft_ms: float
    tpot_ms: float
    total_tokens: int
    total_requests: int


def run_benchmark(model: nn.Module, sequence_length: int, qps: float, duration: float, devices: List[torch.device]) -> BenchmarkResult:
    """Run a synthetic benchmark on ``model``.

    Parameters
    ----------
    model : nn.Module
        The model to test.  It should accept an input tensor of shape
        (batch, seq_len, hidden_dim) and return a tensor of the same
        shape.
    sequence_length : int
        Number of tokens in each request.  All requests in the
        benchmark have the same length when this parameter is used.  To
        explore variable lengths a different benchmark loop can be
        implemented.
    qps : float
        Number of requests per second.  The benchmark will submit
        requests at this rate on average until ``duration`` seconds
        have elapsed.  Requests are processed sequentially in this
        synthetic implementation; in a real system you would leverage
        concurrency and batching to sustain high QPS.
    duration : float
        How long to run the benchmark, in seconds.
    devices : List[torch.device]
        List of devices used by the model.  Metrics are recorded on
        the first device.

    Returns
    -------
    BenchmarkResult
        Aggregated metrics across all processed requests.
    """

    hidden_dim = model.forward(torch.randn(1, 1, 1)).shape[-1]
    # Pre‑generate a batch of random inputs for the given sequence length
    # to reduce overhead during the benchmark loop.
    def generate_request() -> torch.Tensor:
        return torch.randn(1, sequence_length, hidden_dim, device=devices[0])

    total_requests = 0
    total_tokens = 0
    start_time = time.perf_counter()
    end_time = start_time + duration

    ttft_acc = 0.0
    tpot_acc = 0.0

    while time.perf_counter() < end_time:
        # Rate limiting to achieve approximate QPS.  Sleep until the
        # next request should be issued.
        now = time.perf_counter()
        elapsed = now - start_time
        # Target number of requests processed so far
        target_reqs = elapsed * qps
        if total_requests > target_reqs:
            # We're ahead of schedule, sleep a bit
            time.sleep((total_requests - target_reqs) / qps)

        # Generate input and measure TTFT and TPOT.  In this
        # simplified benchmark we define TTFT as the time it takes to
        # produce the first token of the output (which is also the
        # input length).  TPOT is the average per‑token time for the
        # remaining tokens.  Because our model is a feed‑forward layer
        # (no autoregressive generation) the difference between TTFT and
        # TPOT will mostly reflect the cost of the first forward call
        # versus the per‑token cost of copying the output.  In real
        # decoding you would call the model once for the prompt
        # (prefill) and then once per generated token.
        inp = generate_request()
        request_start = time.perf_counter()
        with torch.no_grad():
            out = model(inp)
        request_end = time.perf_counter()
        latency = request_end - request_start
        # For a pure feed‑forward layer we treat the entire latency as
        # the TTFT; there is no autoregressive decoding so TPOT is zero.
        # To approximate TPOT we compute the average time per token
        # assuming the sequence would have been generated one token at a
        # time.  This is a rough proxy and meant only for illustration.
        ttft_acc += latency * 1000.0  # convert to ms
        # Avoid division by zero when sequence_length == 1
        if sequence_length > 1:
            tpot_acc += (latency / (sequence_length - 1)) * 1000.0
        total_requests += 1
        total_tokens += sequence_length

    # Average metrics
    avg_ttft = ttft_acc / total_requests if total_requests else 0.0
    avg_tpot = tpot_acc / total_requests if total_requests else 0.0
    return BenchmarkResult(ttft_ms=avg_ttft, tpot_ms=avg_tpot, total_tokens=total_tokens, total_requests=total_requests)


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(description="Run synthetic MoE benchmarks with different parallelism modes.")
    parser.add_argument(
        "--mode",
        choices=["dp", "tp"],
        required=True,
        help="Parallelism mode: 'dp' for data parallelism or 'tp' for tensor parallelism."
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Dimension of the hidden state (and expert input/output)."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Number of tokens in each request."
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=4.0,
        help="Target requests per second for the benchmark."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How long to run the benchmark (in seconds)."
    )
    parser.add_argument(
        "--max-devices",
        type=int,
        default=8,
        help="Maximum number of devices to use.  If fewer GPUs are available the remainder will run on CPU."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = get_available_devices(args.max_devices)
    print(f"Using devices: {devices}")

    # Instantiate the single expert model and wrap according to the chosen mode
    if args.mode == "dp":
        base_model = SingleExpertMoE(args.hidden_dim)
        model = DataParallelMoE(base_model, devices)
    else:
        model = TensorParallelMoE(args.hidden_dim, devices)

    # Warm‑up pass to initialise weights and caches
    with torch.no_grad():
        dummy = torch.randn(1, args.sequence_length, args.hidden_dim, device=devices[0])
        _ = model(dummy)

    result = run_benchmark(model, args.sequence_length, args.qps, args.duration, devices)
    print("\n===== Benchmark Results =====")
    print(f"Mode           : {'Data Parallel' if args.mode == 'dp' else 'Tensor Parallel'}")
    print(f"Hidden Dim     : {args.hidden_dim}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"QPS            : {args.qps}")
    print(f"Duration (s)   : {args.duration}")
    print(f"Total Requests : {result.total_requests}")
    print(f"Total Tokens   : {result.total_tokens}")
    print(f"Average TTFT   : {result.ttft_ms:.2f} ms")
    print(f"Average TPOT   : {result.tpot_ms:.2f} ms (approximation)")


if __name__ == "__main__":
    main()