"""
mMIMO top-k 안테나 선택 로컬 강건성(local robustness) 검증용 VNNLIB 스펙 생성기.

네트워크는 256차원 입력(16x16 H^T H 행렬을 flatten)을 받아 16차원 출력(안테나별
스코어)을 낸다. "정답"은 데이터셋 라벨이 아니라 원본(섭동 없는) 입력 x0에 대해
네트워크 자신이 고른 top-k(기본 k=8) 선택 결과다.

각 (데이터 포인트 x0, eps)에 대해 다음을 만족하는 VNNLIB 스펙을 만든다:

  입력 제약 : x0_i - eps <= X_i <= x0_i + eps   (L_inf eps-ball, i=0..255)
  출력 제약 : "top-k 선택이 절대 안 바뀐다"의 부정(반증 조건)
              exists i in clean_topk, j not in clean_topk : Y_j >= Y_i
              (즉 top-k 밖의 어떤 출력이 top-k 안의 어떤 출력을 추월/동률)

NeuralSAT 실행 결과:
  unsat -> eps 반경 안 모든 섭동에 대해 top-k 선택이 원본과 정확히 같음이 증명됨 (certified robust)
  sat   -> top-k가 바뀌는 반례(counterexample)가 존재함
  unknown/timeout -> 이 방법으로는 증명하지 못함 (강건하지 않다는 뜻은 아님)

데이터셋(mMIMO_AS_training_data_*.pickle)은 매우 크므로(수 GB), pickle 전체를
메모리에 올리지 않고 필요한 앞부분 행만 memmap으로 읽어온다.
"""

from __future__ import annotations

import argparse
import pathlib
import pickletools
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_NET = (
    SCRIPT_DIR.parent
    / "onnx"
    / "Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID.onnx"
)
# crown/ 저장소(neuralsat과 형제 디렉토리)에 있는 원본 학습 데이터. --data로 재지정 가능.
DEFAULT_DATA = (
    SCRIPT_DIR.parents[3]  # .../neuralsat/src/example/mmimo_topk -> parents[3] = AI_Verification/
    / "crown"
    / "mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle"
)


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


def load_rows(filepath: str, n_rows: int) -> np.ndarray:
    info = peek_ndarray_info(filepath)
    if info.fortran_order or len(info.shape) != 2:
        raise NotImplementedError("2차원 C-order 배열만 지원합니다.")
    mm = np.memmap(filepath, dtype=info.dtype, mode="r", offset=info.data_offset, shape=info.shape)
    return np.array(mm[:n_rows])


def clean_topk(sess: ort.InferenceSession, x0: np.ndarray, k: int) -> tuple[np.ndarray, set[int]]:
    input_name = sess.get_inputs()[0].name
    y0 = sess.run(None, {input_name: x0.astype(np.float32).reshape(1, -1)})[0].reshape(-1)
    topk = set(np.argsort(y0)[-k:].tolist())
    return y0, topk


def write_vnnlib(
    out_path: pathlib.Path,
    x0: np.ndarray,
    eps: float,
    topk: set[int],
    n_out: int,
) -> None:
    n_in = x0.shape[0]
    others = [j for j in range(n_out) if j not in topk]

    lines = [
        f"; mMIMO top-{len(topk)} antenna-selection local robustness spec",
        f"; clean_topk (0-indexed antennas) = {sorted(topk)}",
        f"; eps (L_inf) = {eps}",
        "",
        "; Definition of input variables",
    ]
    lines += [f"(declare-const X_{i} Real)" for i in range(n_in)]
    lines += ["", "; Definition of output variables"]
    lines += [f"(declare-const Y_{i} Real)" for i in range(n_out)]

    lines += ["", "; Input constraints (L_inf eps-ball around x0)"]
    for i in range(n_in):
        lines.append(f"(assert (<= X_{i} {x0[i] + eps:.10f}))")
        lines.append(f"(assert (>= X_{i} {x0[i] - eps:.10f}))")

    lines += ["", '; Output constraints: negation of "clean top-k selection is preserved"']
    lines.append("(assert (or")
    for i in sorted(topk):
        for j in others:
            lines.append(f"\t(and (>= Y_{j} Y_{i}))")
    lines.append("))")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", default=str(DEFAULT_NET), help="Path to the mMIMO ONNX model")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to mMIMO_AS_training_data_*.pickle")
    parser.add_argument("--indices", default="0", help="Comma-separated dataset row indices (e.g. 0,1,2)")
    parser.add_argument(
        "--eps",
        default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1",
        help="Comma-separated L_inf eps values to sweep",
    )
    parser.add_argument("--k", type=int, default=8, help="top-k")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "vnnlib"))
    args = parser.parse_args()

    indices = [int(s) for s in args.indices.split(",")]
    eps_list = [float(s) for s in args.eps.split(",")]

    sess = ort.InferenceSession(args.net, providers=["CPUExecutionProvider"])
    n_out = sess.get_outputs()[0].shape[1]

    X = load_rows(args.data, max(indices) + 1)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        x0 = X[idx]
        y0, topk = clean_topk(sess, x0, args.k)
        print(f"idx={idx} clean_top{args.k}={sorted(topk)}")
        for eps in eps_list:
            out_path = out_dir / f"mmimo_top{args.k}_idx{idx}_eps{eps:g}.vnnlib"
            write_vnnlib(out_path, x0, eps, topk, n_out=n_out)
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
