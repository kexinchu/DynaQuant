#!/usr/bin/env python3
"""
Evaluate Qwen3-30B-A3B accuracy on calibration request datasets.

This script loads a HuggingFace-compatible checkpoint (such as the local
``/workspace/Models/Qwen3-30B-A3B`` directory) and runs deterministic
generation for textual calibration datasets living under
``calibration_datasets/requests``.  It currently supports the following JSONL
datasets whose entries contain ground-truth answers, and it can optionally
wrap user prompts with model-specific reasoning tokens (e.g., Qwen's
``<think>`` tag) to activate richer deliberation modes:

* ``mmlu_pro_200.jsonl`` (multiple-choice, answer key in ``answer``)
* ``mmlu_prox_en_200.jsonl`` (multiple-choice, answer key in ``answer``)
* ``aime25_available_30.jsonl`` (short integer answers in ``answer``)
* ``gpqa_main_200.jsonl`` (open-ended scientific QA with answers in ``reference``)
* ``gsm8k_200.jsonl`` (grade-school math problems expecting numeric answers)
* ``humaneval_200.jsonl`` (Python synthesis tasks validated by unit tests)

Datasets requiring specialised scoring (e.g. LiveBench multimodal tasks or
MultiPL-E code execution) are skipped with an explicit note.

Example
-------
python scripts/evaluate_calibration_accuracy.py \
    --model-path /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
    --dataset-dir calibration_datasets/requests \
    --max-samples 200 \
    --output results_qwen3_30B_int4.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("calibration_eval")


# -----------------------------------------------------------------------------
# Utility data structures
# -----------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Container for per-sample evaluation metadata."""

    prompt_id: Optional[str]
    reference: str
    raw_output: str
    parsed_prediction: Optional[str]
    correct: bool
    notes: Optional[str] = None


@dataclass
class DatasetSummary:
    """Aggregated statistics for one dataset."""

    name: str
    path: Path
    total: int
    correct: int
    skipped: int
    accuracy: Optional[float]
    unsupported: bool = False
    details: Optional[List[SampleResult]] = None


# -----------------------------------------------------------------------------
# Dataset task abstractions
# -----------------------------------------------------------------------------


class BaseTask:
    """Abstract interface for dataset-specific prompting and scoring."""

    def __init__(
        self,
        *,
        name: str,
        input_key: str,
        answer_key: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 16,
    ) -> None:
        self.name = name
        self.input_key = input_key
        self.answer_key = answer_key
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------ prompts
    def build_user_prompt(self, entry: Dict[str, Any]) -> str:
        raise NotImplementedError

    # ---------------------------------------------------------------- reference
    def extract_reference(self, entry: Dict[str, Any]) -> str:
        value = entry.get(self.answer_key)
        if value is None:
            raise KeyError(
                f"Expected key '{self.answer_key}' in dataset entry for task {self.name}"
            )
        return str(value).strip()

    # ----------------------------------------------------------------- parsing
    def parse_prediction(self, raw_output: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (parsed_prediction, notes). Default: return stripped text."""
        parsed = raw_output.strip()
        if not parsed:
            return None, "Empty model output"
        return parsed, None

    # -------------------------------------------------------------- comparison
    def compare(self, prediction: str, reference: str) -> bool:
        return prediction == reference

    # -------------------------------------------------------- comparison notes
    def consume_notes(self) -> Optional[str]:
        """Optional hook for tasks to provide post-comparison notes."""
        return None

    # --------------------------------------------------------- generation args
    def generation_kwargs(self, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge task-specific kwargs with CLI-level defaults."""
        kwargs = dict(base_kwargs)
        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        return kwargs


class MultipleChoiceTask(BaseTask):
    """Multiple-choice questions that expect a single letter response."""

    # Pattern to capture single uppercase option letter (A-J). We will search for
    # the *last* occurrence when parsing to avoid picking up letters in the
    # reasoning.
    _CHOICE_PATTERN = re.compile(r"\b([A-J])\b")

    def __init__(
        self,
        *,
        name: str,
        input_key: str = "prompt",
        answer_key: str = "answer",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 4,
    ) -> None:
        system_prompt = system_prompt or (
            "You are a knowledgeable assistant. Answer multiple-choice questions by "
            "returning only the letter corresponding to the correct option."
        )
        super().__init__(
            name=name,
            input_key=input_key,
            answer_key=answer_key,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def build_user_prompt(self, entry: Dict[str, Any]) -> str:
        question = entry.get(self.input_key)
        if question is None:
            raise KeyError(
                f"Expected key '{self.input_key}' in dataset entry for task {self.name}"
            )
        return (
            f"{question}\n\nRespond with only the letter (A-J) of the correct answer."
        )

    def extract_reference(self, entry: Dict[str, Any]) -> str:
        answer = super().extract_reference(entry)
        answer = answer.strip().upper()
        if not answer or answer[0] not in "ABCDEFGHIJ":
            raise ValueError(f"Invalid multiple-choice answer: {answer!r}")
        return answer[0]

    def parse_prediction(self, raw_output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the model output for multiple‑choice tasks.

        This implementation is robust to chain‑of‑thought reasoning by scanning
        the generated text for a final answer on the last non‑empty line.  It
        will return the last uppercase letter (A–J) on the final line if one
        exists.  Otherwise, it falls back to searching the entire output for
        any option letter.  If none are found, the prediction is treated as
        missing.
        """
        text = raw_output.strip().upper()
        if not text:
            return None, "Empty model output"

        # Split into lines and consider the last non‑empty line first.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            for char in reversed(last):
                if char in "ABCDEFGHIJ":
                    return char, None

        # Fall back: search for the last uppercase option letter in the entire output.
        matches = list(self._CHOICE_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1), None

        return None, f"No option letter found in output: {raw_output!r}"

    def compare(self, prediction: str, reference: str) -> bool:
        return prediction == reference


class NumericAnswerTask(BaseTask):
    """Short integer answers (e.g. AIME, GSM8K)."""

    _NUMBER_PATTERN = re.compile(r"-?\d+")

    def __init__(
        self,
        *,
        name: str,
        input_key: str,
        answer_key: str = "answer",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 8,
    ) -> None:
        system_prompt = system_prompt or (
            "You are a competition math assistant. Solve the problem and reply with "
            "only the integer answer. Do not include explanations."
        )
        super().__init__(
            name=name,
            input_key=input_key,
            answer_key=answer_key,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def build_user_prompt(self, entry: Dict[str, Any]) -> str:
        problem = entry.get(self.input_key)
        if problem is None:
            raise KeyError(
                f"Expected key '{self.input_key}' in dataset entry for task {self.name}"
            )
        return (
            "Provide only the final integer answer to the following problem:\n\n"
            f"{problem}\n\nAnswer:"
        )

    def extract_reference(self, entry: Dict[str, Any]) -> str:
        answer = super().extract_reference(entry)
        if not answer:
            raise ValueError("Empty numeric answer encountered.")
        return answer.strip()

    def parse_prediction(self, raw_output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a numeric answer from the model output.

        This parser is tolerant of chain‑of‑thought reasoning.  It first
        searches for a number following a '####' marker on the last line,
        consistent with GSM8K/AIME formatting.  If none is found, it will
        return the last integer appearing anywhere in the output.  Commas in
        numbers are ignored.  If no integer is discovered, the prediction is
        considered missing.
        """
        # Normalise commas for easier matching.
        text = raw_output.replace(",", "")

        # Look for a '#### <number>' pattern, taking the last match if multiple.
        hash_matches = re.findall(r"####\s*([-+]?\d+)", text)
        if hash_matches:
            # Use the last occurrence to accommodate multiple thoughts.
            candidate = hash_matches[-1].lstrip("+")
            return candidate, None

        # Otherwise extract the last integer in the output.
        matches = self._NUMBER_PATTERN.findall(text)
        if matches:
            candidate = matches[-1].lstrip("+")
            return candidate, None

        return None, f"No integer found in output: {raw_output!r}"

    def compare(self, prediction: str, reference: str) -> bool:
        try:
            return int(prediction) == int(reference)
        except ValueError:
            return prediction.strip() == reference.strip()


class BoxedAnswerTask(BaseTask):
    """Tasks that expect answers enclosed within \\boxed{...} (e.g. GPQA)."""

    _BOX_PATTERN = re.compile(r"\\boxed\s*{([^}]*)}")

    def __init__(
        self,
        *,
        name: str,
        input_key: str = "prompt",
        answer_key: str = "reference",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 64,
    ) -> None:
        system_prompt = system_prompt or (
            "You are a scientific expert. Provide a concise final answer enclosed in "
            "\\boxed{like this}. Include additional reasoning only if necessary, but "
            "ensure the boxed answer appears exactly once."
        )
        super().__init__(
            name=name,
            input_key=input_key,
            answer_key=answer_key,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def build_user_prompt(self, entry: Dict[str, Any]) -> str:
        question = entry.get(self.input_key)
        if question is None:
            raise KeyError(
                f"Expected key '{self.input_key}' in dataset entry for task {self.name}"
            )
        return (
            f"{question}\n\nProvide the final answer enclosed in a single \\boxed{{…}}."
        )

    def extract_reference(self, entry: Dict[str, Any]) -> str:
        reference = super().extract_reference(entry)
        match = self._BOX_PATTERN.search(reference)
        if match:
            return self._normalise(match.group(1))
        # Fallback: use raw reference text.
        return self._normalise(reference)

    def parse_prediction(self, raw_output: str) -> Tuple[Optional[str], Optional[str]]:
        match = self._BOX_PATTERN.search(raw_output)
        if match:
            return self._normalise(match.group(1)), None
        return None, "Model output missing \\boxed{...} answer."

    def compare(self, prediction: str, reference: str) -> bool:
        return self._normalise(prediction) == self._normalise(reference)

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower()


class HumanEvalTask(BaseTask):
    """Evaluate HumanEval problems by executing their canonical unit tests."""

    _CODE_BLOCK_RE = re.compile(
        r"```(?:python)?(.*?)```", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        *,
        name: str = "HumanEval",
        max_new_tokens: int = 512,
        timeout_seconds: float = 15.0,
    ) -> None:
        super().__init__(
            name=name,
            input_key="prompt",
            answer_key="canonical_solution",
            system_prompt=(
                "You are an expert Python programmer. Implement the requested function "
                "so that it passes the provided unit tests. Respond with executable "
                "Python code only, without explanations or commentary."
            ),
            max_new_tokens=max_new_tokens,
        )
        self._timeout_seconds = timeout_seconds
        self._pending_tests: Optional[str] = None
        self._pending_entry_point: Optional[str] = None
        self._last_compare_notes: Optional[str] = None

    def build_user_prompt(self, entry: Dict[str, Any]) -> str:
        prompt = entry.get(self.input_key)
        if prompt is None:
            raise KeyError(
                f"Expected key '{self.input_key}' in dataset entry for task {self.name}"
            )
        return (
            "Complete the following Python function. Return only valid Python code "
            "without any surrounding backticks or explanations.\n\n"
            f"{prompt}"
        )

    def extract_reference(self, entry: Dict[str, Any]) -> str:
        tests = entry.get("tests")
        entry_point = entry.get("entry_point")
        if not tests or not entry_point:
            raise KeyError(
                "HumanEval entries must include 'tests' and 'entry_point'.")
        self._pending_tests = tests
        self._pending_entry_point = entry_point

        reference = entry.get(self.answer_key, "")
        return str(reference).strip()

    def parse_prediction(self, raw_output: str) -> Tuple[Optional[str], Optional[str]]:
        text = raw_output.strip()
        if not text:
            return None, "Empty model output"

        match = self._CODE_BLOCK_RE.search(text)
        if match:
            code = match.group(1).strip()
        else:
            code = text

        if not code:
            return None, "No Python code found in model output"

        entry_point = self._pending_entry_point
        notes = None
        if entry_point and f"def {entry_point}" not in code:
            notes = f"Function definition for '{entry_point}' not detected."

        return code, notes

    def compare(self, prediction: str, reference: str) -> bool:
        tests = self._pending_tests
        entry_point = self._pending_entry_point
        self._pending_tests = None
        self._pending_entry_point = None

        if tests is None or entry_point is None:
            raise RuntimeError(
                "HumanEvalTask.compare called without pending test metadata."
            )

        success, message = self._run_tests(prediction, tests, entry_point)
        self._last_compare_notes = message if not success else None
        return success

    def consume_notes(self) -> Optional[str]:
        notes = self._last_compare_notes
        self._last_compare_notes = None
        return notes

    def _run_tests(
        self, candidate_code: str, tests: str, entry_point: str
    ) -> Tuple[bool, Optional[str]]:
        script = (
            "import sys\n"
            "from typing import *\n\n"
            f"{candidate_code.strip()}\n\n"
            f"{tests.strip()}\n\n"
            "if __name__ == '__main__':\n"
            "    try:\n"
            f"        check({entry_point})\n"
            "    except Exception:\n"
            "        import traceback\n"
            "        traceback.print_exc()\n"
            "        sys.exit(1)\n"
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as handle:
            handle.write(script)
            temp_path = Path(handle.name)

        try:
            completed = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {self._timeout_seconds}s."
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

        if completed.returncode == 0:
            return True, None

        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        message_parts = [part for part in (stderr, stdout) if part]
        message = "\n".join(
            message_parts) if message_parts else "Tests failed."
        return False, message


# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------


# Configure supported tasks with dataset-specific prompting.  Each entry
# overrides the default system prompt to instruct the model to hide its
# reasoning and return the answer in a machine‑parsable format.  See
# accompanying documentation for details on these prompts.
SUPPORTED_TASKS: Dict[str, BaseTask] = {
    # MMLU-Pro and MMLU-ProX-EN are single-answer multiple-choice exams.  We
    # instruct the model to silently choose an option and respond with only
    # the uppercase letter of the correct choice.  No additional explanation
    # should be emitted.
    "mmlu_pro_200.jsonl": MultipleChoiceTask(
        name="MMLU-Pro",
        # Encourage the model to reason but ensure the final answer is clearly
        # indicated on its own line.  Allow more tokens for chain‑of‑thought.
        system_prompt=(
            "You are answering a single‑answer multiple‑choice question. "
            "Think through the problem step by step and feel free to show your reasoning. "
            "At the end of your answer, on a new line, write only the uppercase letter "
            "A, B, C, ..., or J corresponding to the correct option. Do not include "
            "anything else on that final line."
        ),
        max_new_tokens=64,
    ),
    "mmlu_prox_en_200.jsonl": MultipleChoiceTask(
        name="MMLU-ProX-EN",
        system_prompt=(
            "You are answering a single‑answer multiple‑choice question in English. "
            "Reason through the question as needed and show your thinking. "
            "When you finish, on the final line of your response, write only the "
            "uppercase letter (A–J) of the correct answer with no additional text."
        ),
        max_new_tokens=64,
    ),
    # AIME problems expect a three‑digit integer answer.  The system prompt asks
    # the model to think silently and provide its final answer as a three‑digit
    # integer (000–999) on the last line, prefaced with '#### '.  The numeric
    # extractor will still pull the digits from the model’s output.
    "aime25_available_30.jsonl": NumericAnswerTask(
        name="AIME 2025",
        input_key="problem",
        # Use chain‑of‑thought reasoning for challenging math.  Instruct the model
        # to think step by step and present the final answer on the last line
        # preceded by '####'.
        system_prompt=(
            "You are solving an AIME‑style competition math problem. "
            "Please think step by step and explain your reasoning. "
            "On the last line of your response, prefix your final three‑digit "
            "answer with '####'. For example, if the answer is 42 you would write "
            "'#### 042'."
        ),
        max_new_tokens=128,
    ),
    # GPQA questions require a boxed final answer.  Instruct the model to choose
    # the correct option silently and output it enclosed in a single \boxed{} on
    # the last line with no additional commentary.
    "gpqa_main_200.jsonl": BoxedAnswerTask(
        name="GPQA-Main",
        input_key="prompt",
        answer_key="reference",
        # Treat GPQA as a multiple-choice exam.  Permit reasoning and ensure the
        # final answer text is enclosed in a single boxed expression.
        system_prompt=(
            "You are a scientific expert answering a graduate‑level question. "
            "Work through the problem as needed and show your reasoning. "
            "Finish by presenting the final answer enclosed in a single \\boxed{...} "
            "expression that states the answer text exactly once."
        ),
        max_new_tokens=64,
    ),
    # GSM8K problems are grade‑school math questions.  The system prompt directs
    # the model to reason privately and return only the numeric answer on the
    # last line in a '#### <number>' format.  The numeric extractor will still
    # capture the digits even if the prefix is present.
    "gsm8k_200.jsonl": NumericAnswerTask(
        name="GSM8K",
        input_key="question",
        # Enable chain‑of‑thought reasoning and capture the final answer with a
        # '####' marker on the last line.  The numeric parser will pick up the
        # final integer after the marker.
        system_prompt=(
            "You are solving a grade‑school math word problem. "
            "Reason through the problem step by step and show your work. "
            "On the last line of your response, prefix your final numeric answer "
            "with '####'. For example, if the answer is 13 you should output "
            "'#### 13'."
        ),
        max_new_tokens=128,
    ),
    # HumanEval tasks require code generation.  Instantiate the task and then
    # override its system prompt to enforce strictly code‑only responses.  The
    # custom system prompt emphasises that no markdown, no explanations and no
    # comments should be produced.
    "humaneval_200.jsonl": (lambda: (lambda t: (
        setattr(
            t,
            "system_prompt",
            "You are completing a Python function. "
            "Output only valid Python code and nothing else. "
            "Do not include Markdown formatting, comments or explanations. "
            "Keep the provided function signature unchanged and do not print "
            "anything or run tests."
        ),
        t
    )[1])(HumanEvalTask()))(),
}

UNSUPPORTED_FILES: set[str] = set()


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {exc}"
                ) from exc


# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------


def resolve_model_device(model: Any) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return torch.device(model.device)

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        first_device = next(iter(device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)

    try:
        parameter = next(model.parameters())
        return parameter.device
    except StopIteration:
        return torch.device("cpu")


# -----------------------------------------------------------------------------
# Evaluation core
# -----------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_path: str,
    *,
    device: str,
    torch_dtype: Optional[str],
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    LOGGER.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if torch_dtype:
        dtype = getattr(torch, torch_dtype)
    elif device.startswith("cuda"):
        dtype = torch.float16

    LOGGER.info(
        "Loading model from %s (device=%s, dtype=%s)",
        model_path,
        device,
        dtype,
    )
    device_map: Any
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def evaluate_dataset(
    *,
    task: BaseTask,
    path: Path,
    model,
    tokenizer,
    max_samples: Optional[int],
    generation_defaults: Dict[str, Any],
    keep_details: bool,
    user_prefix: str,
    user_suffix: str,
) -> DatasetSummary:
    LOGGER.info("Evaluating dataset %s (%s)", task.name, path.name)

    details: List[SampleResult] = []
    total = 0
    correct = 0
    skipped = 0
    model_device = resolve_model_device(model)

    for entry in iter_jsonl(path):
        if max_samples is not None and total >= max_samples:
            break

        prompt_id = entry.get("id")
        try:
            user_prompt = task.build_user_prompt(entry)
            reference = task.extract_reference(entry)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "Skipping entry %s due to setup error: %s", prompt_id, exc
            )
            skipped += 1
            continue

        messages = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        if user_prefix or user_suffix:
            composed_prompt = f"{user_prefix}{user_prompt}{user_suffix}"
        else:
            composed_prompt = user_prompt

        messages.append({"role": "user", "content": composed_prompt})

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
        )
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        kwargs = task.generation_kwargs(generation_defaults)
        kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = model.generate(**inputs, **kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][prompt_len:]
        raw_output = tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

        parsed_prediction, notes = task.parse_prediction(raw_output)
        is_correct = False
        if parsed_prediction is None:
            skipped += 1
        else:
            try:
                is_correct = task.compare(parsed_prediction, reference)
            except Exception as exc:  # pylint: disable=broad-except
                notes = f"Comparison failed: {exc}"
                skipped += 1
            else:
                compare_notes = task.consume_notes()
                if compare_notes:
                    notes = f"{notes}\n{compare_notes}" if notes else compare_notes
                if is_correct:
                    correct += 1
        total += 1

        if keep_details:
            details.append(
                SampleResult(
                    prompt_id=str(
                        prompt_id) if prompt_id is not None else None,
                    reference=reference,
                    raw_output=raw_output,
                    parsed_prediction=parsed_prediction,
                    correct=is_correct,
                    notes=notes,
                )
            )

    accuracy: Optional[float]
    if total - skipped <= 0:
        accuracy = None
    else:
        accuracy = correct / (total - skipped)

    return DatasetSummary(
        name=task.name,
        path=path,
        total=total,
        correct=correct,
        skipped=skipped,
        accuracy=accuracy,
        unsupported=False,
        details=details if keep_details else None,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-30B-A3B accuracy on calibration datasets."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace-formatted model directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="calibration_datasets/requests",
        help="Directory containing calibration JSONL files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset filenames to evaluate. Defaults to all supported files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on samples per dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save aggregated JSON results.",
    )
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Include per-sample outputs in the JSON results.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (defaults to deterministic).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for all tasks.",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default="",
        help=(
            "Optional string to prepend to every user prompt before sending it to the model. "
            "Useful for activating model-specific reasoning tags such as '<think>'."
        ),
    )
    parser.add_argument(
        "--user-suffix",
        type=str,
        default="",
        help=(
            "Optional string to append to every user prompt before sending it to the model. "
            "Can hold closing reasoning tags like '</think>' or other guard phrases."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device placement for the model (e.g., 'auto', 'cuda:0').",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        help="Optional torch dtype override (e.g., 'bfloat16').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, ...).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        LOGGER.error("Dataset directory not found: %s", dataset_dir)
        return 1

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    if args.datasets:
        selected = args.datasets
    else:
        selected = list(SUPPORTED_TASKS.keys())

    results: Dict[str, Any] = {
        "model": args.model_path,
        "datasets": {},
    }

    sampling_requested = (args.temperature is not None and args.temperature >= 0.0) or (
        args.top_p is not None and args.top_p < 1.0
    )

    generation_defaults: Dict[str, Any] = {"do_sample": sampling_requested}

    if sampling_requested:
        if args.temperature is not None and args.temperature > 0.0:
            generation_defaults["temperature"] = args.temperature
        if args.top_p is not None and args.top_p < 1.0:
            generation_defaults["top_p"] = args.top_p
    if args.max_new_tokens is not None:
        generation_defaults["max_new_tokens"] = args.max_new_tokens

    if args.user_prefix or args.user_suffix:
        LOGGER.info(
            "Applying user prompt wrappers (prefix=%r, suffix=%r) to enable model-specific reasoning modes.",
            args.user_prefix,
            args.user_suffix,
        )

    summaries: List[DatasetSummary] = []

    for dataset_name in selected:
        dataset_path = dataset_dir / dataset_name
        if dataset_name in UNSUPPORTED_FILES:
            LOGGER.warning(
                "Skipping dataset %s (automatic scoring not implemented).",
                dataset_name,
            )
            summaries.append(
                DatasetSummary(
                    name=dataset_name,
                    path=dataset_path,
                    total=0,
                    correct=0,
                    skipped=0,
                    accuracy=None,
                    unsupported=True,
                )
            )
            continue

        task = SUPPORTED_TASKS.get(dataset_name)
        if task is None:
            LOGGER.warning(
                "No task handler registered for %s, skipping.", dataset_name
            )
            summaries.append(
                DatasetSummary(
                    name=dataset_name,
                    path=dataset_path,
                    total=0,
                    correct=0,
                    skipped=0,
                    accuracy=None,
                    unsupported=True,
                )
            )
            continue

        if not dataset_path.exists():
            LOGGER.warning("Dataset file not found: %s", dataset_path)
            summaries.append(
                DatasetSummary(
                    name=task.name,
                    path=dataset_path,
                    total=0,
                    correct=0,
                    skipped=0,
                    accuracy=None,
                    unsupported=True,
                )
            )
            continue

        summary = evaluate_dataset(
            task=task,
            path=dataset_path,
            model=model,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            generation_defaults=generation_defaults,
            keep_details=args.keep_details,
            user_prefix=args.user_prefix,
            user_suffix=args.user_suffix,
        )
        summaries.append(summary)

    for summary in summaries:
        if summary.unsupported:
            LOGGER.info("%s: unsupported (skipped)", summary.name)
        else:
            LOGGER.info(
                "%s: accuracy=%s (%d/%d), skipped=%d",
                summary.name,
                f"{summary.accuracy:.3f}" if summary.accuracy is not None else "n/a",
                summary.correct,
                summary.total - summary.skipped,
                summary.skipped,
            )
        results["datasets"][summary.name] = {
            "path": str(summary.path),
            "total": summary.total,
            "correct": summary.correct,
            "skipped": summary.skipped,
            "accuracy": summary.accuracy,
            "unsupported": summary.unsupported,
        }
        if args.keep_details and summary.details is not None:
            results["datasets"][summary.name]["samples"] = [
                {
                    "id": sample.prompt_id,
                    "reference": sample.reference,
                    "raw_output": sample.raw_output,
                    "parsed_prediction": sample.parsed_prediction,
                    "correct": sample.correct,
                    "notes": sample.notes,
                }
                for sample in summary.details
            ]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        LOGGER.info("Saved evaluation results to %s", output_path)

    LOGGER.info("Evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
