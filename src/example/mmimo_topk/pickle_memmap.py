"""
pickle.dump 된 대용량 numpy 배열을 전체를 메모리에 올리지 않고 다루기 위한 유틸.

원본 pickle 파일(수 GB)의 최상위 ndarray가 pickle protocol 2 이상의
NDArrayWrapper(BINARY / raw 버퍼) 형태로 저장되어 있다는 전제 하에, pickle
opcode 스트림을 훑어 raw 데이터 블록의 (offset, length)만 알아내고, 실제
데이터는 np.memmap 으로 필요한 행 구간만 읽는다.
"""

from __future__ import annotations

import pickletools
from dataclasses import dataclass

import numpy as np


@dataclass
class NdarrayPickleInfo:
    shape: tuple
    dtype: np.dtype
    fortran_order: bool
    data_offset: int
    data_length: int


class _LargeBlobEncountered(Exception):
    def __init__(self, pos: int, n: int):
        self.pos = pos
        self.n = n


class _SkipLargeReads:
    def __init__(self, f, threshold: int):
        self.f = f
        self.threshold = threshold

    def read(self, n: int = -1):
        if isinstance(n, int) and n > self.threshold:
            pos = self.f.tell()
            self.f.seek(n, 1)
            raise _LargeBlobEncountered(pos, n)
        return self.f.read(n)

    def readline(self):
        return self.f.readline()

    def tell(self):
        return self.f.tell()


def peek_ndarray_info(filepath: str, threshold: int = 200_000) -> NdarrayPickleInfo:
    """pickle 파일 최상위 ndarray의 shape/dtype과 raw 데이터 (offset, length)를 raw 바이트를 읽지 않고 알아낸다."""
    shape = None
    dtype = None
    fortran_order = None
    data_offset = None
    data_length = None

    pending_ints: list[int] = []
    last_bool = None
    expect_dtype_str = False

    with open(filepath, "rb") as f:
        wrapper = _SkipLargeReads(f, threshold)
        try:
            for opcode, arg, pos in pickletools.genops(wrapper):
                name = opcode.name
                if name in ("BININT", "BININT1", "BININT2"):
                    pending_ints.append(arg)
                elif name in ("TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"):
                    take = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}.get(name, len(pending_ints))
                    if shape is None and take >= 2 and len(pending_ints) >= take:
                        shape = tuple(pending_ints[-take:])
                    pending_ints.clear()
                elif name == "STACK_GLOBAL":
                    expect_dtype_str = True
                elif name == "SHORT_BINUNICODE":
                    if expect_dtype_str and dtype is None and arg not in ("dtype",):
                        try:
                            dtype = np.dtype(arg)
                        except TypeError:
                            pass
                    expect_dtype_str = False
                elif name in ("NEWTRUE", "NEWFALSE"):
                    last_bool = name == "NEWTRUE"
        except _LargeBlobEncountered as e:
            data_offset = e.pos
            data_length = e.n
            fortran_order = bool(last_bool)

    if shape is None or dtype is None or data_offset is None:
        raise ValueError(f"{filepath}: 최상위 ndarray 구조를 찾지 못했습니다.")

    return NdarrayPickleInfo(shape, dtype, fortran_order, data_offset, data_length)


def load_row_range(filepath: str, start: int, end: int) -> np.ndarray:
    """2차원 C-order 배열의 [start:end) 행만 memmap으로 읽어 온다."""
    info = peek_ndarray_info(filepath)
    if info.fortran_order or len(info.shape) != 2:
        raise NotImplementedError("2차원 C-order 배열만 지원합니다.")
    if not (0 <= start <= end <= info.shape[0]):
        raise ValueError(f"잘못된 행 범위입니다: start={start}, end={end}, total_rows={info.shape[0]}")
    mm = np.memmap(filepath, dtype=info.dtype, mode="r", offset=info.data_offset, shape=info.shape)
    return np.array(mm[start:end])


def load_rows(filepath: str, n_rows: int) -> np.ndarray:
    """앞에서부터 n_rows 개 행만 읽는다 (load_row_range(filepath, 0, n_rows)의 축약형)."""
    return load_row_range(filepath, 0, n_rows)
