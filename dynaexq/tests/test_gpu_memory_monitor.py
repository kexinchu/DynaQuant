from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from dynaexq.experiments.gpu_memory import NvmlProcessMemoryMonitor


class _FakeNvml:
    def __init__(self):
        self.initialized = False
        self.shutdown = False
        self.reads = 0

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.shutdown = True

    @staticmethod
    def nvmlDeviceGetHandleByUUID(uuid):
        assert uuid == "fake-uuid"
        return "handle"

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        assert handle == "handle"
        return SimpleNamespace(total=48_000_000_000)

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle):
        assert handle == "handle"
        self.reads += 1
        return [
            SimpleNamespace(
                pid=os.getpid(),
                usedGpuMemory=1000 + 100 * self.reads,
            ),
            SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=2000),
        ]

    @staticmethod
    def nvmlDeviceGetProcessUtilization(handle, since_timestamp):
        assert handle == "handle"
        assert since_timestamp > 0
        return []


def test_nvml_monitor_records_current_process_high_water(monkeypatch):
    fake = _FakeNvml()
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda index: SimpleNamespace(
            uuid="fake-uuid",
            name="NVIDIA RTX A6000",
        ),
    )
    monitor = NvmlProcessMemoryMonitor(
        [0],
        poll_interval_ms=1.0,
        nvml_module=fake,
    )
    monitor.start()
    time.sleep(0.004)
    result = monitor.stop()
    metadata = monitor.metadata()
    monitor.close()

    assert result["process_hbm_used_baseline_bytes"] == 1100
    assert result["process_hbm_used_peak_bytes"] >= 1200
    assert result["process_hbm_used_peak_delta_bytes"] == (
        result["process_hbm_used_peak_bytes"] - 1100
    )
    assert result["process_hbm_poll_samples"] >= 2
    assert result["foreign_compute_resident_processes_peak"] == 1
    assert result["foreign_hbm_used_peak_bytes"] == 2000
    assert result["foreign_compute_activity_samples"] == 0
    assert result["foreign_sm_util_max_pct"] == 0
    assert result["foreign_mem_util_max_pct"] == 0
    assert result["process_util_poll_samples"] >= 2
    assert metadata == {
        "backend": "nvml",
        "scope": "current_process_selected_device_used_bytes",
        "pid": os.getpid(),
        "poll_interval_ms": 1.0,
        "cuda_device_indices": [0],
        "device_names": ["NVIDIA RTX A6000"],
        "device_uuids": ["fake-uuid"],
        "device_total_bytes": [48_000_000_000],
        "includes_non_pytorch_allocations": True,
        "excludes_other_processes": True,
        "foreign_compute_residency_allowed": True,
        "foreign_compute_activity_policy": (
            "reject_nonzero_nvml_process_utilization"
        ),
        "process_utilization_supported": True,
    }
    assert fake.initialized is True
    assert fake.shutdown is True


def test_nvml_monitor_rejects_foreign_compute_activity(monkeypatch):
    class _ContendedNvml(_FakeNvml):
        def __init__(self):
            super().__init__()
            self.utilization_reads = 0

        def nvmlDeviceGetProcessUtilization(self, handle, since_timestamp):
            self.utilization_reads += 1
            if self.utilization_reads == 1:
                return []
            return [
                SimpleNamespace(
                    pid=os.getpid() + 1,
                    smUtil=10,
                    memUtil=2,
                    timeStamp=since_timestamp + 1,
                )
            ]

    fake = _ContendedNvml()
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda index: SimpleNamespace(
            uuid="fake-uuid",
            name="NVIDIA RTX A6000",
        ),
    )
    monitor = NvmlProcessMemoryMonitor(
        [0],
        poll_interval_ms=1.0,
        nvml_module=fake,
    )
    monitor.start()
    time.sleep(0.003)
    with pytest.raises(RuntimeError, match="NVML polling failed"):
        monitor.stop()
    monitor.close()
