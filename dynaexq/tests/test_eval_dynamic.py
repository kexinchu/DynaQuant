from __future__ import annotations

import subprocess

import pytest

from dynaexq.experiments import eval_dynamic


def test_wait_for_idle_physical_gpu_rechecks_until_idle(
    monkeypatch,
    capsys,
):
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "4096, 75\n", ""),
            subprocess.CompletedProcess([], 0, "5, 0\n", ""),
        ]
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(results)

    monotonic = iter([100.0, 100.0, 130.0])
    monkeypatch.setattr(eval_dynamic.subprocess, "run", fake_run)
    monkeypatch.setattr(eval_dynamic.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(eval_dynamic.time, "sleep", lambda seconds: None)

    eval_dynamic._wait_for_idle_physical_gpu(
        1,
        max_used_memory_mib=1024,
        poll_seconds=30,
    )

    assert len(calls) == 2
    assert "--id=1" in calls[0][0]
    output = capsys.readouterr().out
    assert '"status": "busy"' in output
    assert '"status": "idle"' in output


@pytest.mark.parametrize(
    ("gpu", "memory", "poll"),
    [(-1, 1024, 30), (0, -1, 30), (0, 1024, 0)],
)
def test_wait_for_idle_physical_gpu_rejects_invalid_configuration(
    gpu,
    memory,
    poll,
):
    with pytest.raises(ValueError):
        eval_dynamic._wait_for_idle_physical_gpu(
            gpu,
            max_used_memory_mib=memory,
            poll_seconds=poll,
        )
