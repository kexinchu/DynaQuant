import torch

from dynaexq.runtime.types import Bitwidth, ExpertID
from dynaexq.runtime.weights import DualPrecisionWeights, ParameterIndex


def test_dual_precision_loader(tmp_path):
    w4_state = {
        "embedding.weight": torch.randn(4, 4),
        "layers.0.experts.0.weight": torch.ones(4),
    }
    w2_state = {
        "layers.0.experts.0.weight": torch.ones(4) * 2,
    }
    w4_path = tmp_path / "int4.pt"
    w2_path = tmp_path / "int2.pt"
    torch.save(w4_state, w4_path)
    torch.save(w2_state, w2_path)

    repo = DualPrecisionWeights.from_files(w4_path, w2_path)

    # Non-expert weights prefer higher precision copy.
    non_expert = repo.non_expert_state()
    assert "embedding.weight" in non_expert
    assert torch.equal(non_expert["embedding.weight"],
                       w4_state["embedding.weight"])

    # Experts retain dual precision copies.
    expert = ExpertID(layer=0, idx=0)
    w4_bundle = repo.expert_bundle(expert, Bitwidth.W4)
    w2_bundle = repo.expert_bundle(expert, Bitwidth.W2)
    assert torch.equal(
        w4_bundle.tensors["layers.0.experts.0.weight"],
        w4_state["layers.0.experts.0.weight"],
    )
    assert torch.equal(
        w2_bundle.tensors["layers.0.experts.0.weight"],
        w2_state["layers.0.experts.0.weight"],
    )

    # Parameter index retrieval works for inference bootstrap.
    index = ParameterIndex(
        name="layers.0.experts.0.weight",
        bitwidth=Bitwidth.W4,
        expert=expert,
    )
    tensor = repo.get_tensor(index)
    assert torch.equal(tensor, w4_state["layers.0.experts.0.weight"])

    state_dict = repo.materialize_state_dict()
    assert "layers.0.experts.0.weight" in state_dict
    assert torch.equal(
        state_dict["layers.0.experts.0.weight"], w4_state["layers.0.experts.0.weight"])
