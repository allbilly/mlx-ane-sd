"""Optional per-process Linux scheduling hint for paired M1 benchmarks.

The util-min clamp affects this calling thread and threads it creates. It does
not change the system governor or any other process's scheduling policy.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
import time


class SchedAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("policy", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("nice", ctypes.c_int32),
        ("priority", ctypes.c_uint32),
        ("runtime", ctypes.c_uint64),
        ("deadline", ctypes.c_uint64),
        ("period", ctypes.c_uint64),
        ("util_min", ctypes.c_uint32),
        ("util_max", ctypes.c_uint32),
    ]


def utilization_minimum(value):
    if not 0 <= value <= 1024:
        raise ValueError("CPU utilization minimum must be in 0..1024")
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("This scheduling helper requires arm64 Linux")
    libc = ctypes.CDLL(None, use_errno=True)

    def read():
        attr = SchedAttr(size=ctypes.sizeof(SchedAttr))
        if libc.syscall(275, 0, ctypes.byref(attr), ctypes.sizeof(attr), 0):
            raise OSError(ctypes.get_errno(), "sched_getattr failed")
        return attr

    attr = read()
    before = {name: getattr(attr, name) for name, _ in SchedAttr._fields_}
    # UTIL_CLAMP_MIN | KEEP_POLICY | KEEP_PARAMS leaves the scheduler class,
    # priority, nice and time-slice parameters alone.
    attr.flags |= 0x38
    attr.util_min = value
    if libc.syscall(274, 0, ctypes.byref(attr), 0):
        raise OSError(ctypes.get_errno(), "sched_setattr utilization minimum failed")
    after = read()
    if after.util_min != value:
        raise RuntimeError("Scheduler did not retain the requested utilization minimum")
    return {
        "requested_util_min": value,
        "before": before,
        "after": {name: getattr(after, name) for name, _ in SchedAttr._fields_},
        "scope": "calling thread and subsequently created workers; all benchmark configurations",
    }


def host_observation():
    """Read low-cost Linux observations outside each generation's timer."""
    result = {"monotonic_s": time.monotonic(), "cpu_frequency_khz": {}, "sensors": {}}
    for policy in Path("/sys/devices/system/cpu/cpufreq").glob("policy*"):
        try:
            result["cpu_frequency_khz"][policy.name] = int(
                (policy / "scaling_cur_freq").read_text()
            )
        except (OSError, ValueError):
            pass
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            name = (hwmon / "name").read_text().strip()
        except OSError:
            continue
        if name != "macsmc_hwmon":
            continue
        for sensor in list(hwmon.glob("temp*_input")) + list(
            hwmon.glob("power*_input")
        ):
            try:
                label = (
                    sensor.with_name(sensor.name.removesuffix("_input") + "_label")
                    .read_text()
                    .strip()
                )
                result["sensors"][label] = int(sensor.read_text())
            except (OSError, ValueError):
                pass
    for kind in ("cpu", "memory", "io"):
        try:
            result["pressure_" + kind] = (
                Path("/proc/pressure", kind).read_text().strip()
            )
        except OSError:
            pass
    return result
