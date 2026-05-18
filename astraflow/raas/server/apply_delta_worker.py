"""Subprocess worker that applies a sparse delta to a safetensors file.

Runs as a standalone process (``python -m
astraflow.raas.server.apply_delta_worker <safetensors_path> <delta_path>``)
so that the heavy numpy scatter + mmap page faults cannot hold the GIL
inside the RaaS event-loop process. See
``claude-doc/apply-delta-subprocess-plan.md`` for the full rationale.

Exit codes
----------
- ``0``: delta applied successfully.
- ``2``: usage error (wrong argv).
- ``1``: apply failed; traceback is printed to stderr.
"""

from __future__ import annotations

import mmap as _mmap
import struct as _struct
import sys
import time
import traceback

import numpy as np


def apply_delta(safetensors_path: str, delta_path: str) -> dict:
    """Apply a sparse delta to ``safetensors_path`` in place.

    Delta format (must match ``receiver.apply_delta_and_save`` producer):

        [header 16 bytes: <QHHi num_nonzero element_size flags reserved>]
        [indices: num_nonzero * (8 if flags&1 else 4) bytes]
        [values:  num_nonzero * element_size bytes]

    The safetensors file is mmap'd and patched with a single fancy-indexed
    numpy scatter. On memory-pressured hosts where tmpfs pages can be
    swapped, this is the step that eats seconds per index — which is fine
    here because only this subprocess's GIL is held.

    Returns a timing breakdown dict.
    """
    t0 = time.monotonic()

    with open(delta_path, "rb") as f:
        delta_data = f.read()
    t_read = time.monotonic()

    num_nonzero, element_size, flags, _ = _struct.unpack_from(
        "<QHHi", delta_data, 0,
    )
    use_uint64 = bool(flags & 1)
    idx_dtype = np.uint64 if use_uint64 else np.uint32
    idx_size = 8 if use_uint64 else 4
    header_size = 16

    indices_start = header_size
    indices_end = indices_start + num_nonzero * idx_size
    values_start = indices_end
    values_end = values_start + num_nonzero * element_size

    indices = np.frombuffer(
        delta_data[indices_start:indices_end], dtype=idx_dtype,
    )
    values = np.frombuffer(
        delta_data[values_start:values_end], dtype=np.uint8,
    ).reshape(num_nonzero, element_size)
    t_parse = time.monotonic()

    with open(safetensors_path, "rb") as f:
        sf_header_len = _struct.unpack("<Q", f.read(8))[0]
    data_offset = 8 + sf_header_len

    with open(safetensors_path, "r+b") as f:
        mm = _mmap.mmap(f.fileno(), 0)
        t_mmap = time.monotonic()
        try:
            weight_view = np.ndarray(
                shape=(len(mm) - data_offset,),
                dtype=np.uint8,
                buffer=mm,
                offset=data_offset,
            )
            weight_2d = weight_view.reshape(-1, element_size)
            # Sort indices to convert random scatter into sequential
            # writes — dramatically reduces TLB/cache thrashing on
            # large (15 GB+) mmap'd files.
            sort_order = np.argsort(indices)
            weight_2d[indices[sort_order]] = values[sort_order]
            t_scatter = time.monotonic()
        finally:
            mm.close()

    return {
        "read_delta": t_read - t0,
        "parse": t_parse - t_read,
        "mmap": t_mmap - t_parse,
        "scatter": t_scatter - t_mmap,
        "total": t_scatter - t0,
        "num_nonzero": num_nonzero,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python -m astraflow.raas.server.apply_delta_worker "
            "<safetensors_path> <delta_path>",
            file=sys.stderr,
        )
        return 2

    safetensors_path, delta_path = argv[1], argv[2]
    t_start = time.monotonic()
    try:
        timing = apply_delta(safetensors_path, delta_path)
    except Exception:
        traceback.print_exc()
        return 1
    t_total = time.monotonic() - t_start
    parts = " ".join(f"{k}={v:.3f}s" if isinstance(v, float) else f"{k}={v}"
                     for k, v in timing.items())
    msg = f"apply_delta_worker: wall={t_total:.3f}s {parts}"
    print(msg, flush=True)
    # Also append to a timing log so we can read it without RaaS restart
    try:
        with open("/dev/shm/_delta_timing.log", "a") as f:
            f.write(msg + "\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
