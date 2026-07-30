"""Deterministic quality evaluation for DynaExQ paper artifacts.

The CLI evaluates a *configured checkpoint*.  It does not activate DynaExQ,
ExpertFlow, or a quantization method merely from a label.  Runtime experiments
must configure the model first and call :func:`evaluate`, then record the
result with the same schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from .datasets import EvalRequest, dataset_provenance, load_dataset


SCHEMA_VERSION = 2
PAPER_PROTOCOL = {
    "name": "tc_main_v2",
    "seed": 42,
    "sample_limits": {
        "mmlu_pro": None,
        "gpqa": None,
        "aime25": None,
        "gsm8k": None,
        "humaneval": None,
    },
    "wikitext_max_windows": 128,
    "wikitext_window_tokens": 2048,
}


def _normalize_integer(value: str) -> Optional[str]:
    """Normalize an integer-like answer, rejecting non-integral values."""
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return str(int(number))


def extract_final_integer(text: str) -> Optional[str]:
    """Extract a final integer answer without matching intermediate work."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    patterns = (
        r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*([-+]?\$?[\d,]+(?:\.0+)?)\s*$",
        r"####\s*([-+]?\$?[\d,]+(?:\.0+)?)",
        r"\\boxed\{\s*([-+]?\$?[\d,]+(?:\.0+)?)\s*\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, clean)
        if matches:
            normalized = _normalize_integer(matches[-1])
            if normalized is not None:
                return normalized

    # Fallback to the last numeric token, never "target appears anywhere".
    matches = re.findall(r"(?<![\w.])[-+]?\$?[\d,]+(?:\.0+)?(?![\w.])", clean)
    return _normalize_integer(matches[-1]) if matches else None


def wilson_interval(
    correct: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    if correct < 0 or correct > total:
        raise ValueError("correct must be in [0, total]")
    proportion = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "level": 0.95,
        "low": max(0.0, center - half_width),
        "high": min(1.0, center + half_width),
        "method": "wilson",
    }


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:python)?[ \t]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```[ \t]*", "", text)
    return text.strip("\n\r")


def _limit_child_process() -> None:
    """Apply conservative limits to a HumanEval child process (POSIX)."""
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def execute_humaneval(
    generated: str,
    request: EvalRequest,
    *,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Run the official HumanEval check function in an isolated subprocess."""
    test = str(request.metadata.get("test", ""))
    entry_point = str(request.metadata.get("entry_point", ""))
    if not test or not entry_point:
        return False, "missing_test_metadata"

    code = _strip_code_fences(generated)
    program = (
        request.prompt
        + "\n"
        + code
        + "\n\n"
        + test
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.TemporaryDirectory(prefix="dynaexq-humaneval-") as temp_dir:
        script = Path(temp_dir) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=temp_dir,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limit_child_process if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except OSError as error:
            return False, f"os_error:{type(error).__name__}"
    return (result.returncode == 0, "pass" if result.returncode == 0 else "failed_tests")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _continuation_logprobs(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    continuations: list[str],
    device: torch.device,
) -> list[float]:
    """Score all answer labels in one model state and one unpadded forward."""
    if not continuations:
        raise ValueError("at least one continuation is required")
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids
    continuation_ids = [
        tokenizer(
            continuation,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        for continuation in continuations
    ]
    prompt_len = prompt_ids.shape[1]
    lengths = {ids.shape[1] for ids in continuation_ids}
    if 0 in lengths:
        raise ValueError("answer labels must tokenize to at least one token")
    if len(lengths) != 1:
        raise ValueError(
            "answer labels tokenize to different lengths; padding would "
            "pollute router observations"
        )
    continuation_batch = torch.cat(continuation_ids, dim=0)
    prompt_batch = prompt_ids.repeat(len(continuations), 1)
    full_ids = torch.cat((prompt_batch, continuation_batch), dim=1).to(device)
    with torch.inference_mode():
        logits = model(full_ids).logits
    # logits[:, prompt_len - 1] predicts the first continuation token.
    continuation_logits = logits[:, prompt_len - 1 : -1, :]
    token_logprobs = torch.log_softmax(
        continuation_logits.float(),
        dim=-1,
    ).gather(
        dim=-1,
        index=continuation_batch.to(device).unsqueeze(-1),
    )
    return [
        float(value)
        for value in token_logprobs.squeeze(-1).sum(dim=-1).cpu().tolist()
    ]


def compute_mc_accuracy(
    model: torch.nn.Module,
    tokenizer,
    requests: list[EvalRequest],
    *,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Score complete answer-label continuations instead of one token."""
    model.eval()
    actual_device = torch.device(device) if device else _model_device(model)
    correct = 0
    details = []
    for request in requests:
        n_choices = len(request.choices)
        if n_choices < 2:
            details.append(
                {"sample_id": request.sample_id, "status": "invalid_choices"}
            )
            continue
        letters = [chr(65 + index) for index in range(n_choices)]
        label_scores = _continuation_logprobs(
            model,
            tokenizer,
            request.prompt,
            [f" {letter}" for letter in letters],
            actual_device,
        )
        scores = dict(zip(letters, label_scores))
        prediction = max(scores, key=scores.get)
        is_correct = prediction == str(request.target).strip().upper()
        correct += int(is_correct)
        details.append(
            {
                "sample_id": request.sample_id,
                "prediction": prediction,
                "target": request.target,
                "correct": is_correct,
            }
        )
    total = len(requests)
    evaluated = sum("correct" in item for item in details)
    result = {
        "metric": "accuracy",
        "score": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "evaluated": evaluated,
        "failed": total - evaluated,
        "details": details,
    }
    if total:
        result["confidence_interval"] = wilson_interval(correct, total)
    return result


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    requests: list[EvalRequest],
    *,
    max_length: int = 2048,
    stride: Optional[int] = None,
    max_windows: int = 128,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Token-window perplexity with overlap masking and exact token counts."""
    if not requests:
        return {
            "metric": "perplexity",
            "score": float("nan"),
            "total_nll": 0.0,
            "total_tokens": 0,
            "windows": 0,
            "window_details": [],
        }
    actual_device = torch.device(device) if device else _model_device(model)
    stride = stride or max_length
    corpus = "\n\n".join(request.prompt for request in requests)
    input_ids = tokenizer(corpus, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    windows = 0
    window_details = []
    model.eval()

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - previous_end
        if target_len <= 0:
            continue
        window = input_ids[:, begin:end].to(actual_device)
        labels = window.clone()
        if target_len < labels.shape[1]:
            labels[:, :-target_len] = -100
        with torch.inference_mode():
            output = model(window, labels=labels)
        # Causal LM loss excludes the first label in each window.
        scored = int((labels[:, 1:] != -100).sum().item())
        mean_loss = float(output.loss.item())
        window_nll = mean_loss * scored
        total_nll += window_nll
        total_tokens += scored
        window_details.append(
            {
                "window_index": windows,
                "begin_token": begin,
                "end_token": end,
                "target_tokens": scored,
                "mean_loss": mean_loss,
                "nll": window_nll,
            }
        )
        previous_end = end
        windows += 1
        if end == seq_len or windows >= max_windows:
            break

    score = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
    return {
        "metric": "perplexity",
        "score": score,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "windows": windows,
        "window_tokens": max_length,
        "stride_tokens": stride,
        "window_details": window_details,
    }


def _generate(
    model: torch.nn.Module,
    tokenizer,
    request: EvalRequest,
    device: torch.device,
) -> str:
    encoded = tokenizer(request.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_tokens = encoded["input_ids"].shape[1]
    return tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=True)


def compute_generation_accuracy(
    model: torch.nn.Module,
    tokenizer,
    requests: list[EvalRequest],
    *,
    device: Optional[str] = None,
) -> dict[str, Any]:
    actual_device = torch.device(device) if device else _model_device(model)
    correct = 0
    details = []
    model.eval()
    for request in requests:
        output = _generate(model, tokenizer, request, actual_device)
        prediction = extract_final_integer(output)
        target = _normalize_integer(str(request.target))
        is_correct = prediction is not None and target is not None and prediction == target
        correct += int(is_correct)
        details.append(
            {
                "sample_id": request.sample_id,
                "prediction": prediction,
                "target": target,
                "correct": is_correct,
                "status": "ok" if prediction is not None else "unparsed",
            }
        )
    total = len(requests)
    unparsed = sum(item["status"] != "ok" for item in details)
    result = {
        "metric": "accuracy",
        "score": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        # An unparseable response is a valid evaluated example and scores as
        # incorrect; it is not an infrastructure failure.
        "evaluated": total,
        "failed": 0,
        "unparsed": unparsed,
        "details": details,
    }
    if total:
        result["confidence_interval"] = wilson_interval(correct, total)
    return result


def compute_pass_at_1(
    model: torch.nn.Module,
    tokenizer,
    requests: list[EvalRequest],
    *,
    device: Optional[str] = None,
    allow_code_execution: bool = False,
) -> dict[str, Any]:
    if not allow_code_execution:
        raise RuntimeError(
            "HumanEval executes generated code. Re-run with "
            "--allow-code-execution inside an isolated container."
        )
    actual_device = torch.device(device) if device else _model_device(model)
    correct = 0
    details = []
    model.eval()
    for request in requests:
        output = _generate(model, tokenizer, request, actual_device)
        passed, status = execute_humaneval(output, request)
        correct += int(passed)
        details.append(
            {
                "sample_id": request.sample_id,
                "correct": passed,
                "status": status,
            }
        )
    total = len(requests)
    infrastructure_failures = sum(
        item["status"].startswith(("missing_", "os_error:"))
        for item in details
    )
    result = {
        "metric": "pass@1",
        "score": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "evaluated": total,
        # A candidate that fails an official test (including a timeout) has
        # still been evaluated and simply contributes zero to pass@1.
        "failed": infrastructure_failures,
        "incorrect": total - correct,
        "details": details,
    }
    if total:
        result["confidence_interval"] = wilson_interval(correct, total)
    return result


def evaluate(
    model: torch.nn.Module,
    tokenizer,
    benchmarks: list[str],
    *,
    device: Optional[str] = None,
    n_samples: Optional[int] = None,
    sample_limits: Optional[dict[str, Optional[int]]] = None,
    wikitext_max_windows: int = 128,
    allow_code_execution: bool = False,
) -> dict[str, dict[str, Any]]:
    if n_samples is not None and sample_limits is not None:
        raise ValueError("n_samples and sample_limits are mutually exclusive")
    if wikitext_max_windows <= 0:
        raise ValueError("wikitext_max_windows must be positive")
    results: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        benchmark_limit = (
            sample_limits.get(benchmark)
            if sample_limits is not None
            else n_samples
        )
        if benchmark == "wikitext":
            window_limit = (
                benchmark_limit
                if benchmark_limit is not None
                else wikitext_max_windows
            )
            requests = load_dataset(benchmark, n_samples=window_limit)
        else:
            requests = load_dataset(
                benchmark,
                **(
                    {"n_samples": benchmark_limit}
                    if benchmark_limit is not None
                    else {}
                ),
            )
        started = time.perf_counter()
        if benchmark == "wikitext":
            result = compute_perplexity(
                model,
                tokenizer,
                requests,
                max_windows=window_limit,
                device=device,
            )
        elif benchmark in {"mmlu_pro", "gpqa"}:
            result = compute_mc_accuracy(
                model, tokenizer, requests, device=device
            )
        elif benchmark in {"aime25", "gsm8k"}:
            result = compute_generation_accuracy(
                model, tokenizer, requests, device=device
            )
        elif benchmark == "humaneval":
            result = compute_pass_at_1(
                model,
                tokenizer,
                requests,
                device=device,
                allow_code_execution=allow_code_execution,
            )
        else:
            raise ValueError(f"unsupported benchmark: {benchmark}")
        result["dataset"] = dataset_provenance(requests)
        result["request_limit"] = benchmark_limit
        result["elapsed_s"] = time.perf_counter() - started
        results[benchmark] = result
    return results


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_metadata() -> dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    process_max_rss_bytes = (
        int(max_rss)
        if sys.platform == "darwin"
        else int(max_rss) * 1024
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpu_names,
        "process_max_rss_bytes": process_max_rss_bytes,
        "git": _git_metadata(),
    }


def checkpoint_metadata(
    model_reference: str,
    *,
    hash_weight_files: bool = False,
) -> dict[str, Any]:
    """Describe a local checkpoint without silently hashing huge shards."""
    path = Path(model_reference).expanduser()
    if not path.exists():
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model_reference)
            revision = info.sha
        except Exception as error:
            return {
                "reference": model_reference,
                "local": False,
                "revision": None,
                "resolution_error": type(error).__name__,
                "weight_hashes_included": False,
            }
        return {
            "reference": model_reference,
            "local": False,
            "revision": revision,
            "weight_hashes_included": False,
        }

    def describe(file_path: Path, include_hash: bool) -> dict[str, Any]:
        item: dict[str, Any] = {
            "name": file_path.name,
            "size_bytes": file_path.stat().st_size,
        }
        if include_hash:
            digest = hashlib.sha256()
            with file_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        return item

    if path.is_file():
        return {
            "reference": model_reference,
            "local": True,
            "resolved_path": str(path.resolve()),
            "files": [describe(path, hash_weight_files)],
            "weight_hashes_included": hash_weight_files,
        }

    control_names = {
        "config.json",
        "generation_config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    controls = [
        describe(candidate, True)
        for candidate in sorted(path.iterdir())
        if candidate.is_file() and candidate.name in control_names
    ]
    weights = [
        describe(candidate, hash_weight_files)
        for candidate in sorted(path.iterdir())
        if candidate.is_file()
        and candidate.suffix in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    return {
        "reference": model_reference,
        "local": True,
        "resolved_path": str(path.resolve()),
        "control_files": controls,
        "weight_files": weights,
        "weight_hashes_included": hash_weight_files,
    }


def paper_quality_method(
    paper_model: str,
    method: str,
    quantization: str | None,
) -> str:
    """Return the only manuscript identity allowed for a static quality run."""
    allowed = {
        "qwen30b": {
            ("reference_fp16", None): "reference_fp16",
            ("quantized_checkpoint", "int4"): "static_int4",
        },
        "qwen80b": {
            ("quantized_checkpoint", "int4"): "static_int4",
            ("quantized_checkpoint", "int2"): "static_int2",
        },
        "phi35": {
            ("reference_fp16", None): "reference_fp16",
            ("quantized_checkpoint", "int4"): "static_int4",
        },
    }
    try:
        return allowed[paper_model][(method, quantization)]
    except KeyError as error:
        raise ValueError(
            f"{method}/{quantization} is not a reported {paper_model} "
            "quality method"
        ) from error


def autoround_load_config(model_path: str, backend: str | None):
    """Build an explicit AutoRound inference config from a local checkpoint."""
    if backend is None:
        return None
    config_path = Path(model_path).expanduser().resolve() / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "--autoround-backend requires a local checkpoint with config.json"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config["quantization_config"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("checkpoint has no valid AutoRound config") from error
    method = str(quantization.get("quant_method", "")).replace("_", "-")
    if method != "auto-round":
        raise ValueError("checkpoint is not an AutoRound checkpoint")
    from transformers import AutoRoundConfig

    preserved = dict(quantization)
    for key in ("bits", "group_size", "sym", "backend", "quant_method"):
        preserved.pop(key, None)
    return AutoRoundConfig(
        bits=int(quantization["bits"]),
        group_size=int(quantization["group_size"]),
        sym=bool(quantization["sym"]),
        backend=backend,
        **preserved,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model or local checkpoint")
    parser.add_argument(
        "--paper-model",
        choices=("qwen30b", "qwen80b", "phi35"),
        help="Canonical manuscript model key; required by --paper-protocol",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("reference_fp16", "reference_bf16", "quantized_checkpoint"),
        help="What the CLI can verify from the loaded checkpoint",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=("int2", "int4"),
        help="Required for --method quantized_checkpoint",
    )
    parser.add_argument(
        "--autoround-backend",
        choices=("triton",),
        help=(
            "Explicit AutoRound inference backend. Required for formal "
            "quantized-checkpoint runs to prevent environment-dependent "
            "kernel selection."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Comma-separated benchmark names",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Use the exact per-benchmark sampling protocol reported in the paper",
    )
    parser.add_argument("--allow-code-execution", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-model-files", action="store_true")
    args = parser.parse_args()

    if (args.method == "quantized_checkpoint") != (args.quantization is not None):
        parser.error(
            "--quantization is required exactly when --method=quantized_checkpoint"
        )
    if args.autoround_backend is not None and args.method != "quantized_checkpoint":
        parser.error("--autoround-backend requires --method=quantized_checkpoint")
    if (
        args.paper_protocol
        and args.method == "quantized_checkpoint"
        and args.autoround_backend != "triton"
    ):
        parser.error(
            "formal quantized-checkpoint runs require "
            "--autoround-backend=triton"
        )
    if args.paper_protocol and args.n_samples is not None:
        parser.error("--paper-protocol and --n-samples are mutually exclusive")
    if args.paper_protocol and args.seed != PAPER_PROTOCOL["seed"]:
        parser.error(
            f"--paper-protocol requires --seed={PAPER_PROTOCOL['seed']}"
        )
    if args.paper_protocol and args.paper_model is None:
        parser.error("--paper-protocol requires --paper-model")
    paper_method = None
    if args.paper_model is not None:
        try:
            paper_method = paper_quality_method(
                args.paper_model,
                args.method,
                args.quantization,
            )
        except ValueError as error:
            parser.error(str(error))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.method == "reference_bf16" else torch.float16
    checkpoint = checkpoint_metadata(
        args.model,
        hash_weight_files=args.hash_model_files,
    )
    revision = checkpoint.get("revision")
    if checkpoint.get("local") is False and not revision:
        parser.error(
            "remote checkpoint revision could not be resolved; refusing an "
            "unpinned evaluation"
        )
    if (
        args.paper_protocol
        and checkpoint.get("local") is True
        and not checkpoint.get("weight_hashes_included")
    ):
        parser.error(
            "--paper-protocol requires --hash-model-files for a local checkpoint"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=revision,
        trust_remote_code=False,
    )
    quantization_config = autoround_load_config(
        args.model,
        args.autoround_backend,
    )
    model_kwargs = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=False,
        **model_kwargs,
    )
    model.eval()
    benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    results = evaluate(
        model,
        tokenizer,
        benchmarks,
        device=args.device,
        n_samples=args.n_samples,
        sample_limits=(
            PAPER_PROTOCOL["sample_limits"] if args.paper_protocol else None
        ),
        wikitext_max_windows=(
            PAPER_PROTOCOL["wikitext_max_windows"]
            if args.paper_protocol
            else 128
        ),
        allow_code_execution=args.allow_code_execution,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "quality_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "paper_model": args.paper_model,
        "paper_method": paper_method,
        "checkpoint": checkpoint,
        "method": args.method,
        "quantization": args.quantization,
        "inference_backend": args.autoround_backend,
        "device": args.device,
        "seed": args.seed,
        "evaluation_protocol": (
            PAPER_PROTOCOL if args.paper_protocol else {"name": "custom"}
        ),
        "generation": {"do_sample": False},
        "benchmarks": results,
        "environment": environment_metadata(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({**artifact, "benchmarks": {
        key: {k: v for k, v in value.items() if k != "details"}
        for key, value in results.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
