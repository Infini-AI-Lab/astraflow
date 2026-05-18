"""Byte-level equivalence test for the chunked _compute_delta path.

Runs a whole-buffer (old-style) and a chunked (new-style) sparse delta
over the same pair of uint16 buffers and asserts the emitted delta blob
is byte-identical. Does not import the sender agent; it reimplements the
same logic in small functions so the test is independent of the main
process's state.
"""
import struct

import numpy as np


def _pack_delta(
    num_nonzero: int,
    element_size: int,
    use_uint64: bool,
    indices: np.ndarray,
    values: np.ndarray,
) -> bytes:
    """Common header + body packing used by both impls."""
    header_size = 16
    idx_size = 8 if use_uint64 else 4
    indices_size = num_nonzero * idx_size
    values_size = num_nonzero * element_size
    total_size = header_size + indices_size + values_size
    buf = bytearray(total_size)
    flags = 1 if use_uint64 else 0
    struct.pack_into("<QHHi", buf, 0, num_nonzero, element_size, flags, 0)
    if num_nonzero > 0:
        buf[header_size:header_size + indices_size] = indices.tobytes()
        buf[header_size + indices_size:
            header_size + indices_size + values_size] = values.view(np.uint8).tobytes()
    return bytes(buf)


def whole_buffer_compute(
    new_arr: np.ndarray, old_arr: np.ndarray, force_uint64: bool = False,
) -> bytes:
    """Reference impl: the old path that allocated a single 7.6 GB bool mask."""
    element_size = 2
    num_elements = len(new_arr)
    use_uint64 = force_uint64 or num_elements > (2**32 - 1)
    idx_dtype = np.uint64 if use_uint64 else np.uint32

    diff_mask = new_arr != old_arr
    nonzero_indices = np.where(diff_mask)[0]
    num_nonzero = len(nonzero_indices)
    indices = nonzero_indices.astype(idx_dtype)
    values = new_arr[nonzero_indices]
    return _pack_delta(num_nonzero, element_size, use_uint64, indices, values)


def chunked_compute(
    new_arr: np.ndarray,
    old_arr: np.ndarray,
    chunk_elems: int,
    force_uint64: bool = False,
) -> bytes:
    """New chunked impl matching sender_agent._compute_delta."""
    element_size = 2
    num_elements = len(new_arr)
    use_uint64 = force_uint64 or num_elements > (2**32 - 1)
    idx_dtype = np.uint64 if use_uint64 else np.uint32

    idx_chunks = []
    val_chunks = []
    for start in range(0, num_elements, chunk_elems):
        end = min(start + chunk_elems, num_elements)
        new_chunk = new_arr[start:end]
        old_chunk = old_arr[start:end]
        diff_mask = new_chunk != old_chunk
        local_idx = np.where(diff_mask)[0]
        del diff_mask
        if local_idx.size == 0:
            continue
        global_idx = (local_idx + start).astype(idx_dtype)
        chunk_vals = new_chunk[local_idx]
        idx_chunks.append(global_idx)
        val_chunks.append(chunk_vals)

    if idx_chunks:
        all_idx = np.concatenate(idx_chunks)
        all_vals = np.concatenate(val_chunks)
    else:
        all_idx = np.empty(0, dtype=idx_dtype)
        all_vals = np.empty(0, dtype=np.uint16)

    return _pack_delta(
        int(all_idx.size), element_size, use_uint64, all_idx, all_vals,
    )


def _assert_equiv(new_arr, old_arr, chunk_sizes, force_uint64=False):
    ref = whole_buffer_compute(new_arr, old_arr, force_uint64=force_uint64)
    for chunk in chunk_sizes:
        got = chunked_compute(
            new_arr, old_arr, chunk_elems=chunk, force_uint64=force_uint64,
        )
        assert got == ref, (
            f"mismatch: N={len(new_arr)} chunk={chunk} "
            f"force_uint64={force_uint64} ref_len={len(ref)} got_len={len(got)}"
        )


def test_all_equal():
    N = 10_000
    a = np.arange(N, dtype=np.uint16)
    _assert_equiv(a, a, chunk_sizes=[1, 100, 1024, N, N + 5])
    print("test_all_equal: OK")


def test_all_different():
    N = 10_000
    a = np.zeros(N, dtype=np.uint16)
    b = np.ones(N, dtype=np.uint16)
    _assert_equiv(a, b, chunk_sizes=[1, 100, 1024, N])
    print("test_all_different: OK")


def test_sparse_random():
    rng = np.random.default_rng(42)
    N = 1_000_000
    new_arr = rng.integers(0, 65536, size=N, dtype=np.uint16)
    old_arr = new_arr.copy()
    diff_idx = rng.choice(N, size=N // 200, replace=False)  # ~0.5 % density
    old_arr[diff_idx] = (old_arr[diff_idx].astype(np.int32) + 1).astype(np.uint16)
    _assert_equiv(new_arr, old_arr, chunk_sizes=[1 << 10, 1 << 14, 1 << 17, N])
    print("test_sparse_random: OK")


def test_non_multiple_chunk_boundary():
    rng = np.random.default_rng(7)
    N = 1007  # prime-ish, not a multiple of any round chunk size
    a = rng.integers(0, 65536, size=N, dtype=np.uint16)
    b = a.copy()
    # Sprinkle diffs at irregular positions including near boundaries.
    for i in [0, 1, 99, 100, 101, 255, 256, 257, 500, 1006]:
        b[i] ^= 0xA5A5
    _assert_equiv(a, b, chunk_sizes=[100, 256, 300, 500, 1000, N, N + 1])
    print("test_non_multiple_chunk_boundary: OK")


def test_empty_chunks():
    """Some chunks have zero diffs — ensure concatenation handles it."""
    N = 10_000
    a = np.arange(N, dtype=np.uint16)
    b = a.copy()
    # Only one diff, in the middle — most chunks are empty.
    b[5000] ^= 0xFFFF
    _assert_equiv(a, b, chunk_sizes=[100, 500, 1000, 5000, N])
    print("test_empty_chunks: OK")


def test_force_uint64_path():
    """Exercise the use_uint64 branch even on small arrays."""
    rng = np.random.default_rng(1)
    N = 10_000
    a = rng.integers(0, 65536, size=N, dtype=np.uint16)
    b = a.copy()
    b[::13] ^= 0x1234
    _assert_equiv(a, b, chunk_sizes=[100, 1024, N], force_uint64=True)
    print("test_force_uint64_path: OK")


def test_single_chunk_fits_all():
    """Chunk size >= N: should behave exactly like the whole-buffer path."""
    rng = np.random.default_rng(99)
    N = 5000
    a = rng.integers(0, 65536, size=N, dtype=np.uint16)
    b = a.copy()
    b[::7] += 1
    _assert_equiv(a, b, chunk_sizes=[N, N + 1, 10 * N])
    print("test_single_chunk_fits_all: OK")


if __name__ == "__main__":
    test_all_equal()
    test_all_different()
    test_empty_chunks()
    test_non_multiple_chunk_boundary()
    test_sparse_random()
    test_force_uint64_path()
    test_single_chunk_fits_all()
    print("ALL PASS")
