import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_model_registry_references_existing_artifacts_and_manifests():
    registry = json.loads(
        (ROOT / "release" / "model_registry.json").read_text(encoding="utf-8")
    )
    artifacts = registry["artifacts"]
    for model in registry["paper_models"].values():
        for binding in model["methods"].values():
            artifact_id = binding.get("artifact") or binding.get("source_artifact")
            assert artifact_id in artifacts
    for artifact in artifacts.values():
        if "manifest" in artifact:
            assert (ROOT / artifact["manifest"]).is_file()


def test_project_release_revisions_match_huggingface_manifest():
    registry = json.loads(
        (ROOT / "release" / "model_registry.json").read_text(encoding="utf-8")
    )
    release = json.loads(
        (ROOT / "release" / "huggingface" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    published = {
        model["repository"]: model for model in release["models"]
    }
    for artifact in registry["artifacts"].values():
        if artifact["origin"] != "dynaexq_release":
            continue
        model = published[artifact["repository"]]
        assert artifact["revision"] == model["revision"]
        assert artifact["expected_safetensors_bytes"] == model["safetensors_bytes"]
