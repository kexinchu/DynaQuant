#!/usr/bin/env python3
"""Fail closed unless the current manuscript is ready for IEEE TC upload."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "ICCAD_2026_DynExq"
MAIN_TEX = PAPER_DIR / "main_sc.tex"
MAIN_PDF = PAPER_DIR / "main_sc.pdf"
MAIN_LOG = PAPER_DIR / "main_sc.log"
METADATA = PAPER_DIR / "TC_SUBMISSION_METADATA.json"
RESULT_AUDIT = ROOT / "scripts" / "audit_paper_results.py"

VENUE = "IEEE Transactions on Computers"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-[\dX]{4}$")
PLACEHOLDER_RE = re.compile(
    r"\b(?:TBD|TODO|PLACEHOLDER)\b|"
    r"Anonymous Authors|"
    r"Internal draft|"
    r"not submission-ready|"
    r"Registered performance artifact pending",
    re.IGNORECASE,
)


def _inside_root(path_text: str, *, must_exist: bool = True) -> Path:
    relative = Path(path_text)
    if relative.is_absolute():
        raise ValueError("path must be repository-relative")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("path escapes the repository")
    if must_exist and not path.is_file():
        raise ValueError(f"file does not exist: {relative}")
    return path


def _valid_orcid(value: str) -> bool:
    if not ORCID_RE.fullmatch(value):
        return False
    compact = value.replace("-", "")
    if compact[:15] == "0" * 15:
        return False
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    check = (12 - total % 11) % 11
    expected = "X" if check == 10 else str(check)
    return compact[-1] == expected


def _validate_metadata(data: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["INVALID TC METADATA: root must be an object"]
    if data.get("venue") != VENUE:
        problems.append(f"INVALID TC VENUE: expected {VENUE!r}")
    if data.get("article_type") != "regular_paper":
        problems.append("INVALID TC ARTICLE TYPE: expected regular_paper")

    review_model = data.get("review_model")
    if review_model not in {"single_anonymous", "double_anonymous_requested"}:
        problems.append("INVALID TC REVIEW MODEL")

    authors = data.get("authors")
    author_emails: set[str] = set()
    if not isinstance(authors, list) or not authors:
        problems.append("MISSING TC AUTHORS")
    else:
        for index, author in enumerate(authors):
            if not isinstance(author, dict):
                problems.append(f"INVALID TC AUTHOR: authors[{index}]")
                continue
            name = str(author.get("name", "")).strip()
            affiliation = str(author.get("affiliation", "")).strip()
            email = str(author.get("email", "")).strip()
            orcid = str(author.get("orcid", "")).strip()
            if (
                len(name.split()) < 2
                or "anonymous" in name.lower()
                or not affiliation
                or not EMAIL_RE.fullmatch(email)
                or email.endswith("@example.edu")
            ):
                problems.append(f"INVALID TC AUTHOR: authors[{index}]")
            elif not _valid_orcid(orcid):
                problems.append(
                    f"MISSING OR INVALID TC AUTHOR ORCID: {name}"
                )
            if email:
                if email in author_emails:
                    problems.append(f"DUPLICATE TC AUTHOR EMAIL: {email}")
                author_emails.add(email)

    corresponding = str(data.get("corresponding_author_email", "")).strip()
    if not EMAIL_RE.fullmatch(corresponding) or corresponding not in author_emails:
        problems.append("INVALID TC CORRESPONDING AUTHOR")

    prior = data.get("prior_version")
    if not isinstance(prior, dict):
        problems.append("MISSING PRIOR-VERSION DECLARATION")
        return problems
    status = prior.get("status")
    if status == "under_review_elsewhere":
        problems.append("CONCURRENT SUBMISSION IS NOT ALLOWED")
    elif status in {"none", "rejected_unpublished"}:
        declaration = str(prior.get("declaration", "")).strip()
        if len(declaration) < 20:
            problems.append("INCOMPLETE NO-PRIOR-VERSION DECLARATION")
        if status == "rejected_unpublished":
            previous = prior.get("previous_submission")
            if (
                not isinstance(previous, dict)
                or not str(previous.get("venue", "")).strip()
                or previous.get("outcome") != "rejected"
                or previous.get("published") is not False
            ):
                problems.append(
                    "INCOMPLETE REJECTED-UNPUBLISHED SUBMISSION RECORD"
                )
    elif status in {"published_conference", "planned_conference"}:
        for field in ("title", "citation", "paper_path", "difference_statement_path"):
            if not str(prior.get(field, "")).strip():
                problems.append(f"INCOMPLETE PRIOR VERSION FIELD: {field}")
        for field in ("paper_path", "difference_statement_path"):
            path_text = str(prior.get(field, "")).strip()
            if path_text:
                try:
                    _inside_root(path_text)
                except ValueError as error:
                    problems.append(f"INVALID PRIOR VERSION {field}: {error}")
        if status == "published_conference":
            try:
                new_pct = float(prior["estimated_new_material_pct"])
                similarity_pct = float(
                    prior["estimated_verbatim_similarity_pct"]
                )
            except (KeyError, TypeError, ValueError):
                problems.append("MISSING EXTENDED-VERSION PERCENTAGES")
            else:
                if new_pct < 40.0:
                    problems.append("INSUFFICIENT NEW MATERIAL: requires >=40%")
                if not 0.0 <= similarity_pct < 30.0:
                    problems.append(
                        "EXCESSIVE VERBATIM SIMILARITY: requires <30%"
                    )
    else:
        problems.append("INVALID PRIOR-VERSION STATUS")

    disclosure = data.get("human_or_animal_research")
    if not isinstance(disclosure, bool):
        problems.append("MISSING HUMAN/ANIMAL RESEARCH DECLARATION")
    return problems


def _extract_environment(text: str, name: str) -> str | None:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}",
        text,
        re.DOTALL,
    )
    return match.group(1) if match else None


def _validate_tex(text: str, metadata: dict[str, Any] | None) -> list[str]:
    problems: list[str] = []
    active_lines = [
        line.split("%", 1)[0].strip()
        for line in text.splitlines()
        if line.split("%", 1)[0].strip()
    ]
    if r"\artifactverifiedtrue" not in active_lines:
        problems.append("RESULT GATE IS NOT VERIFIED")
    if r"\artifactverifiedfalse" in active_lines:
        problems.append("DRAFT RESULT GATE IS ACTIVE")
    if r"\documentclass[journal]{IEEEtran}" not in active_lines:
        problems.append("INVALID IEEE JOURNAL DOCUMENT CLASS")

    abstract = _extract_environment(text, "abstract")
    if abstract is None:
        problems.append("MISSING ABSTRACT")
    else:
        plain = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", abstract)
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", plain)
        if not 100 <= len(words) <= 200:
            problems.append(
                f"INVALID ABSTRACT LENGTH: {len(words)} words; expected 100--200"
            )
        if re.search(r"\\cite|\\ref|\$|\\\(|\\\[", abstract):
            problems.append("ABSTRACT CONTAINS CITATION, REFERENCE, OR MATH")

    keywords = _extract_environment(text, "IEEEkeywords")
    if keywords is None or not keywords.strip():
        problems.append("MISSING IEEE KEYWORDS")

    review_model = metadata.get("review_model") if metadata else None
    author_start = text.find(r"\author{")
    document_start = text.find(r"\begin{document}")
    author_text = (
        text[author_start:document_start]
        if 0 <= author_start < document_start
        else ""
    )
    if review_model == "single_anonymous":
        if not author_text or "anonymous" in author_text.lower():
            problems.append("SINGLE-ANONYMOUS SUBMISSION LACKS AUTHORS")
        for author in metadata.get("authors", []):
            surname = str(author.get("name", "")).strip().split()[-1:]
            if surname and surname[0] not in author_text:
                problems.append(
                    f"MANUSCRIPT AUTHOR MISMATCH: {author.get('name')}"
                )
    elif review_model == "double_anonymous_requested":
        if "anonymous" not in author_text.lower():
            problems.append("DOUBLE-ANONYMOUS MANUSCRIPT EXPOSES AUTHORS")
    return problems


def _validate_rendered_text(text: str) -> list[str]:
    matches = sorted({match.group(0) for match in PLACEHOLDER_RE.finditer(text)})
    return [f"RENDERED MANUSCRIPT CONTAINS PLACEHOLDER: {value}" for value in matches]


def _validate_pdfinfo(text: str, size_bytes: int) -> list[str]:
    problems: list[str] = []
    pages_match = re.search(r"^Pages:\s+(\d+)", text, re.MULTILINE)
    if not pages_match:
        problems.append("PDF PAGE COUNT IS UNAVAILABLE")
    elif int(pages_match.group(1)) > 12:
        problems.append(f"TC PAGE LIMIT EXCEEDED: {pages_match.group(1)} > 12")
    size_match = re.search(r"^Page size:\s+(.+)$", text, re.MULTILINE)
    if not size_match or "letter" not in size_match.group(1).lower():
        problems.append("PDF IS NOT US LETTER")
    if size_bytes > 350 * 1024**2:
        problems.append("PDF EXCEEDS 350 MB")
    return problems


def _validate_log(text: str) -> list[str]:
    patterns = {
        "LATEX OVERFULL BOX": r"Overfull \\[hv]box",
        "LATEX UNDEFINED REFERENCE": r"(?:Citation|Reference).+undefined",
        "LATEX UNDEFINED REFERENCES": r"There were undefined references",
        "LATEX RERUN REQUIRED": r"Rerun to get cross-references right",
    }
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def audit_submission(*, run_result_audit: bool = True) -> list[str]:
    problems: list[str] = []
    metadata: dict[str, Any] | None = None
    if not METADATA.is_file():
        problems.append(f"MISSING TC SUBMISSION METADATA: {METADATA}")
    else:
        try:
            loaded = json.loads(METADATA.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            problems.append(f"INVALID TC SUBMISSION METADATA: {error}")
        else:
            problems.extend(_validate_metadata(loaded))
            if isinstance(loaded, dict):
                metadata = loaded

    if not MAIN_TEX.is_file():
        problems.append(f"MISSING MANUSCRIPT SOURCE: {MAIN_TEX}")
    else:
        problems.extend(
            _validate_tex(MAIN_TEX.read_text(encoding="utf-8"), metadata)
        )

    if not MAIN_PDF.is_file():
        problems.append(f"MISSING MANUSCRIPT PDF: {MAIN_PDF}")
    else:
        info = _run(["pdfinfo", str(MAIN_PDF)])
        if info.returncode != 0:
            problems.append("PDFINFO FAILED")
        else:
            problems.extend(
                _validate_pdfinfo(info.stdout, MAIN_PDF.stat().st_size)
            )
        rendered = _run(["pdftotext", str(MAIN_PDF), "-"])
        if rendered.returncode != 0:
            problems.append("PDFTOTEXT FAILED")
        else:
            problems.extend(_validate_rendered_text(rendered.stdout))

    if not MAIN_LOG.is_file():
        problems.append(f"MISSING FINAL LATEX LOG: {MAIN_LOG}")
    else:
        problems.extend(_validate_log(MAIN_LOG.read_text(encoding="utf-8")))

    if run_result_audit:
        result = _run([sys.executable, str(RESULT_AUDIT)])
        if result.returncode != 0:
            problems.append("RESULT PROVENANCE AUDIT FAILED")
            problems.extend(
                f"RESULT AUDIT: {line.removeprefix('- ').strip()}"
                for line in result.stdout.splitlines()
                if line.startswith("- ")
            )
    return problems


def main() -> int:
    problems = audit_submission()
    if problems:
        print("IEEE TC submission audit FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("IEEE TC submission audit PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
