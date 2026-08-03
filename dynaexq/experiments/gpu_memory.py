"""Whole-process GPU-memory high-water monitoring through NVML.

PyTorch allocator counters omit memory owned by CUDA extensions and external
runtimes.  Formal paper runs therefore poll NVML's per-process accounting for
the selected physical GPU while each generation is in flight.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import torch


class NvmlProcessMemoryMonitor:
    """Poll current-process HBM use for one or more CUDA devices."""

    def __init__(
        self,
        device_indices: list[int],
        *,
        poll_interval_ms: float = 2.0,
        nvml_module: Any | None = None,
    ) -> None:
        if not device_indices:
            raise ValueError("at least one CUDA device is required")
        if poll_interval_ms <= 0.0 or poll_interval_ms > 10.0:
            raise ValueError("poll_interval_ms must be in (0, 10]")
        if nvml_module is None:
            try:
                import pynvml as nvml_module
            except ImportError as error:
                raise RuntimeError(
                    "formal GPU-memory monitoring requires nvidia-ml-py"
                ) from error

        self._nvml = nvml_module
        self._nvml.nvmlInit()
        self._closed = False
        self._active = False
        self._interval_s = poll_interval_ms / 1000.0
        self._pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._poll_error: BaseException | None = None
        self._baseline_bytes = 0
        self._peak_bytes = 0
        self._sample_count = 0
        self._foreign_resident_processes_peak = 0
        self._foreign_hbm_used_peak_bytes = 0
        self._foreign_compute_activity_samples = 0
        self._foreign_sm_util_max_pct = 0
        self._foreign_mem_util_max_pct = 0
        self._process_util_poll_samples = 0

        self._device_indices = [int(index) for index in device_indices]
        self._device_names: list[str] = []
        self._device_uuids: list[str] = []
        self._handles = []
        self._device_total_bytes: list[int] = []
        for index in self._device_indices:
            if index < 0 or index >= torch.cuda.device_count():
                self.close()
                raise ValueError(f"invalid CUDA device index: {index}")
            properties = torch.cuda.get_device_properties(index)
            uuid = str(getattr(properties, "uuid", ""))
            if not uuid:
                self.close()
                raise RuntimeError(
                    f"CUDA device {index} exposes no stable UUID"
                )
            try:
                handle = self._nvml.nvmlDeviceGetHandleByUUID(uuid)
                total_bytes = int(
                    self._nvml.nvmlDeviceGetMemoryInfo(handle).total
                )
            except Exception:
                self.close()
                raise
            self._device_uuids.append(uuid)
            self._device_names.append(str(properties.name))
            self._handles.append(handle)
            self._device_total_bytes.append(total_bytes)
        self._process_utilization_method = getattr(
            self._nvml,
            "nvmlDeviceGetProcessUtilization",
            None,
        )
        if not callable(self._process_utilization_method):
            self.close()
            raise RuntimeError(
                "formal shared-GPU measurement requires NVML per-process "
                "utilization accounting"
            )
        self._process_util_since_us = [0 for _ in self._handles]

    def _processes(self, handle: Any) -> list[Any]:
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            method = getattr(self._nvml, name, None)
            if callable(method):
                return list(method(handle))
        raise RuntimeError(
            "NVML per-process GPU-memory accounting is unavailable"
        )

    @staticmethod
    def _used_gpu_memory(process: Any) -> int:
        used = getattr(process, "usedGpuMemory", None)
        if isinstance(used, bool) or not isinstance(used, int) or used < 0:
            raise RuntimeError(
                "NVML returned unavailable per-process GPU memory"
            )
        return used

    def _read_process_bytes(self) -> tuple[int, int, int]:
        total = 0
        foreign_count = 0
        foreign_bytes = 0
        for handle in self._handles:
            processes = self._processes(handle)
            matches = [
                process
                for process in processes
                if int(getattr(process, "pid", -1)) == self._pid
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    "NVML returned duplicate records for the current process"
                )
            if matches:
                total += self._used_gpu_memory(matches[0])
            for process in processes:
                pid = int(getattr(process, "pid", -1))
                if pid < 0:
                    raise RuntimeError("NVML returned an unavailable process ID")
                if pid == self._pid:
                    continue
                foreign_count += 1
                foreign_bytes += self._used_gpu_memory(process)
        return total, foreign_count, foreign_bytes

    @staticmethod
    def _util_value(sample: Any, name: str) -> int:
        value = getattr(sample, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"NVML returned unavailable per-process {name}"
            )
        return value

    def _read_process_utilization(self) -> None:
        foreign_activity = 0
        for index, handle in enumerate(self._handles):
            try:
                samples = list(
                    self._process_utilization_method(
                        handle,
                        self._process_util_since_us[index],
                    )
                )
            except Exception as error:
                # NVML reports "not found" when no process-utilization sample
                # exists in the requested interval.
                if error.__class__.__name__ == "NVMLError_NotFound":
                    samples = []
                else:
                    raise
            newest_timestamp = self._process_util_since_us[index]
            for sample in samples:
                timestamp = self._util_value(sample, "timeStamp")
                newest_timestamp = max(newest_timestamp, timestamp + 1)
                pid = self._util_value(sample, "pid")
                if pid == self._pid:
                    continue
                sm_util = self._util_value(sample, "smUtil")
                mem_util = self._util_value(sample, "memUtil")
                self._foreign_sm_util_max_pct = max(
                    self._foreign_sm_util_max_pct,
                    sm_util,
                )
                self._foreign_mem_util_max_pct = max(
                    self._foreign_mem_util_max_pct,
                    mem_util,
                )
                if sm_util > 0 or mem_util > 0:
                    foreign_activity += 1
            self._process_util_since_us[index] = newest_timestamp
        self._process_util_poll_samples += 1
        self._foreign_compute_activity_samples += foreign_activity
        if foreign_activity:
            raise RuntimeError(
                "formal GPU measurement observed nonzero foreign-process "
                "compute or memory utilization"
            )

    def _record_sample(self) -> None:
        used, foreign_count, foreign_bytes = self._read_process_bytes()
        self._peak_bytes = max(self._peak_bytes, used)
        self._foreign_resident_processes_peak = max(
            self._foreign_resident_processes_peak,
            foreign_count,
        )
        self._foreign_hbm_used_peak_bytes = max(
            self._foreign_hbm_used_peak_bytes,
            foreign_bytes,
        )
        self._sample_count += 1
        self._read_process_utilization()

    def _poll(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._record_sample()
            except BaseException as error:
                self._poll_error = error
                self._stop.set()
                return

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("NVML monitor is closed")
        if self._active:
            raise RuntimeError("NVML monitor is already active")
        self._stop.clear()
        self._poll_error = None
        self._process_util_since_us = [
            time.time_ns() // 1_000 for _ in self._handles
        ]
        (
            self._baseline_bytes,
            foreign_count,
            foreign_bytes,
        ) = self._read_process_bytes()
        self._peak_bytes = self._baseline_bytes
        self._sample_count = 0
        self._foreign_resident_processes_peak = foreign_count
        self._foreign_hbm_used_peak_bytes = foreign_bytes
        self._foreign_compute_activity_samples = 0
        self._foreign_sm_util_max_pct = 0
        self._foreign_mem_util_max_pct = 0
        self._process_util_poll_samples = 0
        self._record_sample()
        self._active = True
        self._thread = threading.Thread(
            target=self._poll,
            name="dynaexq-nvml-memory",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, int]:
        if not self._active:
            raise RuntimeError("NVML monitor is not active")
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._thread = None
        self._active = False
        if self._poll_error is not None:
            raise RuntimeError("NVML polling failed") from self._poll_error
        self._record_sample()
        return {
            "process_hbm_used_baseline_bytes": self._baseline_bytes,
            "process_hbm_used_peak_bytes": self._peak_bytes,
            "process_hbm_used_peak_delta_bytes": (
                self._peak_bytes - self._baseline_bytes
            ),
            "process_hbm_poll_samples": self._sample_count,
            "foreign_compute_resident_processes_peak": (
                self._foreign_resident_processes_peak
            ),
            "foreign_hbm_used_peak_bytes": self._foreign_hbm_used_peak_bytes,
            "foreign_compute_activity_samples": (
                self._foreign_compute_activity_samples
            ),
            "foreign_sm_util_max_pct": self._foreign_sm_util_max_pct,
            "foreign_mem_util_max_pct": self._foreign_mem_util_max_pct,
            "process_util_poll_samples": self._process_util_poll_samples,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "nvml",
            "scope": "current_process_selected_device_used_bytes",
            "pid": self._pid,
            "poll_interval_ms": self._interval_s * 1000.0,
            "cuda_device_indices": list(self._device_indices),
            "device_names": list(self._device_names),
            "device_uuids": list(self._device_uuids),
            "device_total_bytes": list(self._device_total_bytes),
            "includes_non_pytorch_allocations": True,
            "excludes_other_processes": True,
            "foreign_compute_residency_allowed": True,
            "foreign_compute_activity_policy": (
                "reject_nonzero_nvml_process_utilization"
            ),
            "process_utilization_supported": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._active:
            try:
                self.stop()
            except Exception:
                pass
        self._nvml.nvmlShutdown()
        self._closed = True

    def __enter__(self) -> "NvmlProcessMemoryMonitor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
