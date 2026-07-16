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
메모리에 올리지 않고 필요한 행만 memmap으로 읽어온다.

입력 데이터는 data_split.py 기준으로 train/test 로 분할된 것 중 **test 구간**만
사용한다 (--indices는 test 구간 내 상대 인덱스, 0 = test 구간 첫 샘플). 학습에
쓰인 train 구간은 절대 사용하지 않는다.

이 스크립트는 실제 .vnnlib 텍스트 파일을 디스크에 쓰지 않는다 (인스턴스당
~26KB, 최종 목표인 28만 인스턴스 기준 수GB + 파일 수십만 개로 쌓여 공간/파일수
낭비가 심하기 때문). 대신 인스턴스별로 (idx, eps, net, data, k, abs_row)만
manifest.csv에 기록해 두고, 실제 VNNLIB 텍스트는 run_batch.py가 NeuralSAT을
호출하기 직전에 임시 파일로 즉석 생성했다가 실행 후 바로 삭제한다.

manifest.csv는 idx별 정확한 clean_topk 계산을 실행 시점(run_batch.py)에 다시
하므로, 이 스크립트에서의 clean_topk 계산은 유효성 확인/미리보기 용도다.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import onnxruntime as ort
from data_split import test_row_range
from pickle_memmap import load_row_range

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_NET = (
    SCRIPT_DIR.parent
    / "onnx"
    / "Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID.onnx"
)
DEFAULT_DATA = pathlib.Path(
    r"C:\AI_Verification\wireless\Pickle\mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "vnnlib"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

MANIFEST_FIELDS = ["idx", "eps", "net", "data", "k", "abs_row", "result"]


def clean_topk(sess: ort.InferenceSession, x0: np.ndarray, k: int) -> tuple[np.ndarray, set[int]]:
    input_name = sess.get_inputs()[0].name
    y0 = sess.run(None, {input_name: x0.astype(np.float32).reshape(1, -1)})[0].reshape(-1)
    topk = set(np.argsort(y0)[-k:].tolist())
    return y0, topk


def build_vnnlib_text(x0: np.ndarray, eps: float, topk: set[int], n_out: int) -> str:
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

    return "\n".join(lines) + "\n"


def result_path_for(results_dir: pathlib.Path, k: int, idx: int, eps: float) -> pathlib.Path:
    return results_dir / f"mmimo_top{k}_idx{idx}_eps{eps:g}.result"


def merge_manifest(
    manifest_path: pathlib.Path,
    new_rows: list[dict],
) -> list[dict]:
    """기존 manifest.csv 행(특히 이미 result가 있는 행)은 그대로 유지하면서,
    새로 요청된 (idx, eps) 조합만 추가한다. (더 이상 out_dir을 스캔하지 않는다.)
    """
    existing: dict[tuple[int, float], dict] = {}
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (int(row["idx"]), float(row["eps"]))
                existing[key] = row

    for row in new_rows:
        key = (row["idx"], row["eps"])
        existing.setdefault(key, {
            "idx": str(row["idx"]),
            "eps": repr(row["eps"]),
            "net": row["net"],
            "data": row["data"],
            "k": str(row["k"]),
            "abs_row": str(row["abs_row"]),
            "result": row["result"],
        })

    merged = sorted(existing.values(), key=lambda r: (int(r["idx"]), float(r["eps"])))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(merged)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", default=str(DEFAULT_NET), help="Path to the mMIMO ONNX model")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to mMIMO_AS_training_data_*.pickle")
    parser.add_argument(
        "--indices",
        default="0",
        help="Comma-separated row indices relative to the test split (0 = first test-split sample). "
        "Ignored if --num-samples is given.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Generate specs for N test-split samples starting at --idx-start (default: indices 0..N-1). "
        "Overrides --indices. Use this to scale up incrementally toward the full test split, or to give "
        "each machine a disjoint idx range (e.g. --idx-start 20000 --num-samples 20000).",
    )
    parser.add_argument(
        "--idx-start",
        type=int,
        default=0,
        help="First idx to generate when --num-samples is given (idx = idx-start .. idx-start+num-samples-1).",
    )
    parser.add_argument(
        "--eps",
        default="1e-6,1e-5,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.1,1.0",
        help="Comma-separated L_inf eps values to sweep",
    )
    parser.add_argument("--k", type=int, default=8, help="top-k")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where manifest.csv is kept")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Where run_batch.py will write results")
    parser.add_argument("--no-data-in-file", type=int, default=20000, help="Samples per original source file")
    parser.add_argument("--no-test-files", type=int, default=2, help="Number of trailing files reserved for test")
    args = parser.parse_args()

    if args.num_samples is not None:
        indices = list(range(args.idx_start, args.idx_start + args.num_samples))
    else:
        indices = [int(s) for s in args.indices.split(",")]
    eps_list = [float(s) for s in args.eps.split(",")]

    sess = ort.InferenceSession(args.net, providers=["CPUExecutionProvider"])
    n_out = sess.get_outputs()[0].shape[1]

    test_start, test_end = test_row_range(args.data, args.no_data_in_file, args.no_test_files)
    n_test = test_end - test_start
    if max(indices) >= n_test or min(indices) < 0:
        raise ValueError(f"--indices must be within the test split range [0, {n_test}), got range end {max(indices)}")

    # 실제로 필요한 [min(indices), max(indices)] 구간만 memmap으로 읽는다.
    row_lo, row_hi = min(indices), max(indices)
    X = load_row_range(args.data, test_start + row_lo, test_start + row_hi + 1)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_dir)

    new_rows = []
    for idx in indices:
        x0 = X[idx - row_lo]
        abs_row = test_start + idx
        y0, topk = clean_topk(sess, x0, args.k)
        print(f"idx={idx} (abs row {abs_row}) clean_top{args.k}={sorted(topk)}")
        for eps in eps_list:
            new_rows.append({
                "idx": idx,
                "eps": eps,
                "net": args.net,
                "data": args.data,
                "k": args.k,
                "abs_row": abs_row,
                "result": str(result_path_for(results_dir, args.k, idx, eps)),
            })

    manifest_path = out_dir / "manifest.csv"
    n_before = 0
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            n_before = sum(1 for _ in csv.DictReader(f))
    merged = merge_manifest(manifest_path, new_rows)
    print(f"manifest now has {len(merged)} instances ({len(merged) - n_before} newly added)")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
