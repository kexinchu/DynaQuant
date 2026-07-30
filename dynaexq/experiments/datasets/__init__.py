"""
Benchmark dataset loaders for DynaExq evaluation (Phase 6, plan §7.2).

Each loader returns a list of ``EvalRequest`` objects with a unified
interface that the evaluation scripts consume directly.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Optional


DATASET_REVISIONS = {
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "math-ai/aime25": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
}


@dataclass
class EvalRequest:
    """One evaluation sample."""

    prompt: str
    target: Optional[str] = None
    max_new_tokens: int = 256
    task_type: str = "generation"  # "ppl" | "mc" | "generation" | "code"
    sample_id: str = ""
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _dataset_metadata(
    dataset,
    *,
    repository: str,
    config: Optional[str],
    split: str,
) -> dict[str, Any]:
    """Return an immutable dataset identity suitable for result artifacts."""
    return {
        "repository": repository,
        "revision": DATASET_REVISIONS[repository],
        "config": config,
        "split": split,
        "source_rows": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def dataset_provenance(requests: list[EvalRequest]) -> dict[str, Any]:
    """Extract and validate the common dataset identity of a request set."""
    identities = [
        request.metadata.get("dataset")
        for request in requests
        if request.metadata.get("dataset") is not None
    ]
    if not requests or len(identities) != len(requests):
        raise ValueError("every evaluation request must carry dataset provenance")
    identity = identities[0]
    if any(candidate != identity for candidate in identities[1:]):
        raise ValueError("evaluation requests contain mixed dataset provenance")
    return {**identity, "evaluated_rows": len(requests)}


def load_dataset(name: str, **kwargs) -> list[EvalRequest]:
    """Dispatch to the appropriate loader by name."""
    loaders = {
        "wikitext": _load_wikitext,
        "mmlu_pro": _load_mmlu_pro,
        "gpqa": _load_gpqa,
        "aime25": _load_aime25,
        "gsm8k": _load_gsm8k,
        "humaneval": _load_humaneval,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(loaders)}")
    return loaders[name](**kwargs)


def _load_wikitext(
    split: str = "test",
    n_samples: int = 128,
    seq_len: int = 2048,
) -> list[EvalRequest]:
    """WikiText-2 for perplexity evaluation."""
    from datasets import load_dataset as hf_load

    repository = "Salesforce/wikitext"
    config = "wikitext-2-raw-v1"
    ds = hf_load(
        repository,
        config,
        split=split,
        revision=DATASET_REVISIONS[repository],
    )
    identity = _dataset_metadata(
        ds, repository=repository, config=config, split=split
    )
    # Do not slice here: ``seq_len`` is a model-token count, not a character
    # count.  The evaluator tokenizes the corpus and creates exact token
    # windows.  Keeping one corpus request also avoids silently changing the
    # sample boundaries when tokenizers differ.
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return [
        EvalRequest(
            prompt=text,
            target=text,
            max_new_tokens=0,
            task_type="ppl",
            sample_id=f"wikitext-2-raw-v1/{split}",
            metadata={
                "n_windows": n_samples,
                "window_tokens": seq_len,
                "dataset": identity,
            },
        )
    ]


def _load_mmlu_pro(
    split: str = "test",
    n_samples: Optional[int] = None,
) -> list[EvalRequest]:
    """MMLU-Pro multiple-choice evaluation."""
    from datasets import load_dataset as hf_load

    repository = "TIGER-Lab/MMLU-Pro"
    ds = hf_load(
        repository,
        split=split,
        revision=DATASET_REVISIONS[repository],
    )
    identity = _dataset_metadata(
        ds, repository=repository, config=None, split=split
    )
    requests = []
    for item in ds:
        options = item.get("options", [])
        question = item.get("question", "")
        option_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )
        prompt = f"Question: {question}\n{option_text}\nAnswer:"
        target = str(item.get("answer", "")).strip().upper()
        sample_id = str(item.get("question_id", item.get("id", len(requests))))
        requests.append(
            EvalRequest(
                prompt=prompt,
                target=target,
                max_new_tokens=1,
                task_type="mc",
                sample_id=f"mmlu_pro/{sample_id}",
                choices=tuple(str(option) for option in options),
                metadata={"dataset": identity},
            )
        )
        if n_samples and len(requests) >= n_samples:
            break
    return requests


def _load_gpqa(
    split: str = "train",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[EvalRequest]:
    """GPQA diamond multiple-choice evaluation."""
    from datasets import load_dataset as hf_load

    repository = "Idavidrein/gpqa"
    config = "gpqa_diamond"
    ds = hf_load(
        repository,
        config,
        split=split,
        revision=DATASET_REVISIONS[repository],
    )
    identity = _dataset_metadata(
        ds, repository=repository, config=config, split=split
    )
    requests = []
    for index, item in enumerate(ds):
        question = item.get("Question", "")
        correct = str(item.get("Correct Answer", "")).strip()
        choices = [correct]
        choices.extend(
            str(item.get(f"Incorrect Answer {i}", "")).strip()
            for i in range(1, 4)
        )
        if not question.strip() or not correct or any(not choice for choice in choices):
            continue

        # GPQA stores the correct answer first.  Leaving that order intact
        # makes an "always A" classifier perfect.  Shuffle every sample with
        # a deterministic, per-sample seed so all methods see the same order.
        digest = hashlib.sha256(f"{seed}:{index}:{question}".encode()).digest()
        sample_rng = random.Random(int.from_bytes(digest[:8], "big"))
        sample_rng.shuffle(choices)
        target = chr(65 + choices.index(correct))
        option_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt = f"Question: {question}\n{option_text}\nAnswer:"
        requests.append(
            EvalRequest(
                prompt=prompt,
                target=target,
                max_new_tokens=1,
                task_type="mc",
                sample_id=f"gpqa_diamond/{index}",
                choices=tuple(choices),
                metadata={"shuffle_seed": seed, "dataset": identity},
            )
        )
        if n_samples and len(requests) >= n_samples:
            break
    return requests


def _load_aime25(n_samples: Optional[int] = None) -> list[EvalRequest]:
    """AIME 2025 math problems — numerical answer extraction."""
    from datasets import load_dataset as hf_load

    repository = "math-ai/aime25"
    split = "test"
    ds = hf_load(
        repository,
        split=split,
        revision=DATASET_REVISIONS[repository],
    )
    if len(ds) != 30:
        raise ValueError(f"AIME25 source must contain exactly 30 rows, found {len(ds)}")
    identity = _dataset_metadata(
        ds, repository=repository, config=None, split=split
    )
    return _aime25_requests(ds, identity, n_samples)


def _aime25_requests(
    rows,
    identity: dict[str, Any],
    n_samples: Optional[int] = None,
) -> list[EvalRequest]:
    """Validate AIME25 rows and convert them to evaluation requests."""
    requests = []
    seen_ids: set[str] = set()
    for item in rows:
        problem = str(item.get("problem", "")).strip()
        answer = str(item.get("answer", "")).strip()
        sample_id = str(item.get("id", len(requests)))
        if not problem:
            raise ValueError(f"AIME25 row {sample_id} has no problem text")
        if not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise ValueError(f"AIME25 row {sample_id} has invalid answer {answer!r}")
        if sample_id in seen_ids:
            raise ValueError(f"AIME25 contains duplicate id {sample_id!r}")
        seen_ids.add(sample_id)
        prompt = (
            "Solve the following math problem. Give only the final numerical "
            f"answer.\n\n{problem}\n\nAnswer:"
        )
        requests.append(
            EvalRequest(
                prompt=prompt,
                target=answer,
                max_new_tokens=32,
                task_type="generation",
                sample_id=f"aime25/{sample_id}",
                metadata={"dataset": identity},
            )
        )
        if n_samples and len(requests) >= n_samples:
            break
    return requests


def _load_gsm8k(
    split: str = "test",
    n_samples: Optional[int] = None,
) -> list[EvalRequest]:
    """GSM8K grade-school math — numerical string match."""
    from datasets import load_dataset as hf_load

    repository = "openai/gsm8k"
    config = "main"
    ds = hf_load(
        repository,
        config,
        split=split,
        revision=DATASET_REVISIONS[repository],
    )
    identity = _dataset_metadata(
        ds, repository=repository, config=config, split=split
    )
    requests = []
    for item in ds:
        question = item.get("question", "")
        answer_text = item.get("answer", "")
        final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
        prompt = f"Question: {question}\nAnswer: Let's think step by step."
        requests.append(
            EvalRequest(
                prompt=prompt,
                target=final_answer,
                max_new_tokens=256,
                task_type="generation",
                sample_id=f"gsm8k/{item.get('id', len(requests))}",
                metadata={"dataset": identity},
            )
        )
        if n_samples and len(requests) >= n_samples:
            break
    return requests


def _load_humaneval(n_samples: Optional[int] = None) -> list[EvalRequest]:
    """HumanEval code generation — pass@1."""
    from datasets import load_dataset as hf_load

    repository = "openai/openai_humaneval"
    split = "test"
    ds = hf_load(
        repository,
        split=split,
        revision=DATASET_REVISIONS[repository],
    )
    identity = _dataset_metadata(
        ds, repository=repository, config=None, split=split
    )
    requests = []
    for item in ds:
        prompt = item.get("prompt", "")
        canonical = item.get("canonical_solution", "")
        test = item.get("test", "")
        requests.append(
            EvalRequest(
                prompt=prompt,
                target=canonical,
                max_new_tokens=256,
                task_type="code",
                sample_id=str(item.get("task_id", f"humaneval/{len(requests)}")),
                metadata={
                    "test": test,
                    "entry_point": item.get("entry_point", ""),
                    "dataset": identity,
                },
            )
        )
        if n_samples and len(requests) >= n_samples:
            break
    return requests


__all__ = ["DATASET_REVISIONS", "EvalRequest", "dataset_provenance", "load_dataset"]
