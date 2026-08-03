"""
AutoRound-backed quantization path (Phase 2.1).

This is the live implementation of the ``backend="autoround"`` seam in
:mod:`dynaexq.core.quant`. It calls into
``auto_round.data_type.get_quant_func`` to produce a quantized weight and
then repacks the result into our :class:`PackedTensor` byte layout so the
rest of the runtime (BudgetTracker, TransitionEngine, fused_linear) never
has to know which backend produced the tensor.

What this module DOES right now
-------------------------------
* Calls ``auto_round.data_type.get_quant_func('int', bits=N, sym=True)``,
  which returns an RTN-style symmetric int quantizer that accepts an
  optional per-input-channel ``imatrix`` (importance matrix).
* Derives ``q_signed`` from the dequantized output + per-group scales,
  then uses the existing ``_pack_int4`` / ``_pack_int2`` helpers from
  :mod:`quant` to produce a byte stream with **identical layout** to the
  reference backend.
* Returns a ``PackedTensor`` with ``fmt``, ``group_size``, ``nbytes``,
  and all other invariants matching what the reference backend would
  return for the same shape — so a ``BudgetTracker.try_reserve`` is
  numerically identical whether the tensor came from reference or
  autoround.

What this module does NOT do yet (deferred to Phase 6)
------------------------------------------------------
* It does not run the full AutoRound optimization loop (``WrapperLinear``
  + sign-SGD tuning). That loop requires real activation inputs from a
  forward pass through the model, which we don't have until Phase 6
  wires up eval runs. When that lands, this module gains a
  ``pack_with_autoround_optimized()`` entry point and the seam in
  ``quant.pack()`` routes to it when calibration samples are provided.
* The current `rtn_int_sym` / `int_sym_gptq` entry points accept an
  ``imatrix`` argument but ignore it in the per-tensor call (imatrix
  matters inside the calibration loop only). We still compute and pass
  it through so the call site is forward compatible.

So today the **numerical** output of ``backend="autoround"`` is
equivalent to the reference backend modulo minor fp16 rounding from
auto-round's internal fp32→fp16 promotions. The **value** of having this
seam live is:

1. All the byte accounting, fence logic, and BudgetTracker machinery is
   exercised against an external library call — so when Phase 6 swaps
   in the optimized path, it's a one-line change inside this file
   rather than a runtime refactor.
2. Phase 6 can collect activation samples and call
   ``pack_with_autoround(weight, fmt, calibration_inputs=samples)``
   directly, and we don't have to retrofit the API.
"""

from __future__ import annotations

from typing import Optional

import torch

from .quant import (
    DEFAULT_GROUP_SIZE,
    PackedTensor,
    QuantFormat,
    _pack_int2,
    _pack_int4,
    _qrange,
    compute_packed_nbytes,
)


def _compute_imatrix(calibration_inputs: torch.Tensor) -> torch.Tensor:
    """
    Derive a per-input-channel importance vector from calibration samples.

    Args:
        calibration_inputs: ``(n_samples, in_features)`` float tensor of
            activation vectors collected from the linear's input side.
            Multi-dim inputs (e.g. ``(batch, seq, in)``) are flattened to
            2-D first.

    Returns:
        fp32 tensor of shape ``(in_features,)`` — the RMS of activations
        per input channel. This is the standard imatrix used by
        GPTQ / AWQ style calibration: channels that see large
        activations are quantized with more care.
    """
    x = calibration_inputs
    if x.dim() < 2:
        raise ValueError(
            f"calibration_inputs must have at least 2 dims, got shape {tuple(x.shape)}"
        )
    x = x.reshape(-1, x.shape[-1]).to(torch.float32)
    # Standard imatrix: sqrt(mean(x^2)) per channel.
    return torch.sqrt((x * x).mean(dim=0) + 1e-12)


def pack_with_autoround(
    weight: torch.Tensor,
    fmt: QuantFormat,
    group_size: Optional[int] = None,
    calibration_inputs: Optional[torch.Tensor] = None,
) -> PackedTensor:
    """
    Quantize ``weight`` via auto-round and return a :class:`PackedTensor`.

    Args:
        weight: ``(out_features, in_features)`` float tensor.
        fmt: ``QuantFormat.INT4`` or ``QuantFormat.INT2``. ``FP16`` is not
            a valid target here — the reference backend handles it.
        group_size: Group size along input dim; defaults to
            ``DEFAULT_GROUP_SIZE[fmt]`` (128 for int4, 64 for int2).
        calibration_inputs: Optional activation samples for imatrix
            computation. Passed through to auto-round; today's RTN entry
            point ignores it, but the seam is future-proof.

    Returns:
        PackedTensor with the same byte layout as the reference backend.
    """
    try:
        from auto_round.data_type import get_quant_func
    except ImportError as e:
        raise RuntimeError(
            "auto_round is not installed. Run `pip install auto-round` or "
            "use backend='reference'."
        ) from e

    if weight.dim() != 2:
        raise ValueError(
            f"pack_with_autoround expects a 2D weight, got shape {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise ValueError(
            f"weight must be a float tensor, got dtype {weight.dtype}"
        )
    if fmt == QuantFormat.FP16:
        raise ValueError(
            "pack_with_autoround does not handle FP16; the reference backend's "
            "pack() is a no-op copy for fp16, use that instead"
        )
    if fmt not in (QuantFormat.INT4, QuantFormat.INT2):
        raise ValueError(f"Unsupported QuantFormat for autoround backend: {fmt}")

    out_features, in_features = weight.shape
    if group_size is None:
        group_size = DEFAULT_GROUP_SIZE[fmt]
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )

    # Resolve the auto-round quantizer. ``get_quant_func('int', bits, sym=True)``
    # returns the registered symmetric int quantizer — today that's
    # ``rtn_int_sym``; it may be upgraded to a GPTQ-style variant in a
    # future auto-round release without requiring us to change call site.
    fn, _ = get_quant_func("int", bits=fmt.nbits, sym=True)

    imatrix = None
    if calibration_inputs is not None:
        imatrix_vec = _compute_imatrix(calibration_inputs)
        if imatrix_vec.numel() != in_features:
            raise ValueError(
                f"calibration_inputs last dim {imatrix_vec.numel()} does not "
                f"match weight in_features {in_features}"
            )
        imatrix = imatrix_vec

    # Run auto-round's per-tensor quantizer. Returns (dequant, scales, zp).
    # ``dequant`` has the same shape as ``weight``; ``scales`` comes back
    # flattened to ``(out_features * n_groups, 1)``.
    weight_fp16 = weight.to(torch.float16).contiguous()
    dequant, scales_flat, _zp = fn(
        weight_fp16,
        bits=fmt.nbits,
        group_size=group_size,
        imatrix=imatrix,
    )

    n_groups = in_features // group_size
    # Reshape scales to our ``(out, n_groups)`` convention. Note:
    # auto-round may return **signed** scales (e.g. a negative scale for
    # a mostly-positive group — see ``rtn_int_sym`` which uses
    # ``scale = -max(|w|)/|qmin|`` so the max-abs element lands exactly
    # on qmin). We must preserve the sign verbatim because
    # ``dequant_to_fp16`` reconstructs ``w = q * scale``, and flipping
    # the scale sign would flip every reconstructed value.
    scales = scales_flat.reshape(out_features, n_groups).contiguous().to(torch.float16)

    # Recover the signed integer quantization levels from the
    # dequantized tensor. Because dequant = q * scale, we can divide and
    # round to recover q. Guard against an exact-zero scale (dead group)
    # without clobbering the sign — use ``where`` with a safe fallback
    # of 1.0 which produces q=0 for that group (matching a zero
    # dequantized tensor).
    qmin, qmax = _qrange(fmt.nbits)
    dq_grouped = dequant.to(torch.float32).view(out_features, n_groups, group_size)
    scales_fp32 = scales.to(torch.float32).unsqueeze(-1)
    safe_scales = torch.where(
        scales_fp32.abs() < 1e-12,
        torch.ones_like(scales_fp32),
        scales_fp32,
    )
    q_signed = (
        (dq_grouped / safe_scales).round().clamp(qmin, qmax).to(torch.int8)
    )
    q_signed = q_signed.view(out_features, in_features)

    if fmt == QuantFormat.INT4:
        qweight = _pack_int4(q_signed)
    else:
        qweight = _pack_int2(q_signed)

    return PackedTensor(
        qweight=qweight,
        scales=scales,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
        fmt=fmt,
        nbytes=compute_packed_nbytes(out_features, in_features, fmt, group_size),
    )


__all__ = ["pack_with_autoround"]
