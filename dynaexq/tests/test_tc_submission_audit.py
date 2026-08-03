from __future__ import annotations

from scripts.audit_tc_submission import (
    _validate_log,
    _validate_metadata,
    _validate_pdfinfo,
    _validate_rendered_text,
    _validate_tex,
    _valid_orcid,
)


def _metadata() -> dict:
    return {
        "venue": "IEEE Transactions on Computers",
        "article_type": "regular_paper",
        "review_model": "single_anonymous",
        "authors": [
            {
                "name": "Ada Lovelace",
                "affiliation": "Analytical Engine Institute",
                "email": "ada@institute.test",
                "orcid": "0000-0002-1825-0097",
            }
        ],
        "corresponding_author_email": "ada@institute.test",
        "prior_version": {
            "status": "none",
            "declaration": (
                "No portion of this manuscript is published or under "
                "review elsewhere."
            ),
        },
        "human_or_animal_research": False,
    }


def _tex(*, verified: bool = True, anonymous: bool = False) -> str:
    abstract = " ".join(f"word{index}" for index in range(120))
    gate = r"\artifactverifiedtrue" if verified else r"\artifactverifiedfalse"
    author = "Anonymous Authors" if anonymous else "Ada Lovelace"
    return rf"""
\documentclass[journal]{{IEEEtran}}
{gate}
\author{{{author}}}
\begin{{document}}
\begin{{abstract}}
{abstract}
\end{{abstract}}
\begin{{IEEEkeywords}}
runtime systems, memory management
\end{{IEEEkeywords}}
\end{{document}}
"""


def test_orcid_checksum_and_submission_metadata():
    assert _valid_orcid("0000-0002-1825-0097")
    assert not _valid_orcid("0000-0002-1825-0098")
    assert not _valid_orcid("0000-0000-0000-0000")
    assert _validate_metadata(_metadata()) == []


def test_metadata_rejects_concurrent_and_underextended_versions():
    metadata = _metadata()
    metadata["prior_version"] = {"status": "under_review_elsewhere"}
    assert "CONCURRENT SUBMISSION IS NOT ALLOWED" in _validate_metadata(metadata)

    metadata["prior_version"] = {
        "status": "published_conference",
        "title": "Prior paper",
        "citation": "Complete citation",
        "paper_path": "missing-prior.pdf",
        "difference_statement_path": "missing-differences.md",
        "estimated_new_material_pct": 39,
        "estimated_verbatim_similarity_pct": 30,
    }
    problems = _validate_metadata(metadata)
    assert "INSUFFICIENT NEW MATERIAL: requires >=40%" in problems
    assert "EXCESSIVE VERBATIM SIMILARITY: requires <30%" in problems


def test_metadata_accepts_rejected_unpublished_submission():
    metadata = _metadata()
    metadata["prior_version"] = {
        "status": "rejected_unpublished",
        "declaration": (
            "This manuscript was submitted to ICCAD and rejected; it was "
            "not published and is not under review elsewhere."
        ),
        "previous_submission": {
            "venue": "ICCAD",
            "outcome": "rejected",
            "published": False,
        },
    }
    assert _validate_metadata(metadata) == []


def test_tex_gate_abstract_and_authors():
    metadata = _metadata()
    assert _validate_tex(_tex(), metadata) == []

    problems = _validate_tex(_tex(verified=False, anonymous=True), metadata)
    assert "RESULT GATE IS NOT VERIFIED" in problems
    assert "DRAFT RESULT GATE IS ACTIVE" in problems
    assert "SINGLE-ANONYMOUS SUBMISSION LACKS AUTHORS" in problems


def test_rendered_pdf_and_latex_checks():
    problems = _validate_rendered_text(
        "Anonymous Authors\nPeak HBM: TBD\nnot submission-ready"
    )
    assert len(problems) == 3
    assert _validate_rendered_text("Complete camera-ready manuscript") == []

    assert _validate_pdfinfo(
        "Pages:           12\nPage size:       612 x 792 pts (letter)\n",
        1024,
    ) == []
    problems = _validate_pdfinfo(
        "Pages:           13\nPage size:       595 x 842 pts (A4)\n",
        351 * 1024**2,
    )
    assert "TC PAGE LIMIT EXCEEDED: 13 > 12" in problems
    assert "PDF IS NOT US LETTER" in problems
    assert "PDF EXCEEDS 350 MB" in problems

    assert _validate_log("Output written on main_sc.pdf") == []
    assert "LATEX OVERFULL BOX" in _validate_log(
        "Overfull \\hbox (1.0pt too wide)"
    )
