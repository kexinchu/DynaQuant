"""Byte-lease accounting and deadline replay for DynaByte.

The decision layer is pure Python so it can be tested without a GPU and driven
by measured routing traces. The executor in ``dynaexq.core`` remains the
device-side reserve/copy/publish path.
"""

from dynaexq.lease.exchange import Exchange, admit_exchanges, counterfactual_gain
from dynaexq.lease.ledger import ByteLease, LeaseLedger
from dynaexq.lease.replay import LayerResult, simulate_forward

__all__ = [
    "ByteLease",
    "Exchange",
    "LayerResult",
    "LeaseLedger",
    "admit_exchanges",
    "counterfactual_gain",
    "simulate_forward",
]
