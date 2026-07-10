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

최종적으로는 test 구간 전체(4만개)를 대상으로 하지만, 지금은 --num-samples로
원하는 개수만큼만 점진적으로 생성한다. 이미 생성된 .vnnlib은 다시 만들지 않고,
실행 시마다 out-dir에 있는 모든 .vnnlib을 훑어 manifest.csv(run_batch.py /
summarize_results.py가 읽는 인스턴스 목록)를 다시 만든다.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re

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
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

VNNLIB_NAME_RE = re.compile(r"^mmimo_top(?P<k>\d+)_idx(?P<idx>\d+)_eps(?P<eps>.+)\.vnnlib$")


def rebuild_manifest(out_dir: pathlib.Path, results_dir: pathlib.Path, net: str, k: int) -> pathlib.Path:
    """out_dir에 있는 (net에 해당하는 k의) 모든 .vnnlib을 훑어 manifest.csv를 다시 만든다.

    run_batch.py / summarize_results.py는 이 manifest.csv 하나만 보고 동작하므로,
    --num-samples로 점진적으로 늘려가며 여러 번 generate_vnnlib.py를 돌려도 항상
    디스크에 있는 파일들과 일치하는 최신 인스턴스 목록을 유지한다.
    """
    rows = []
    for path in out_dir.glob("*.vnnlib"):
        m = VNNLIB_NAME_RE.match(path.name)
        if m is None or int(m.group("k")) != k:
            continue
        idx = int(m.group("idx"))
        eps = m.group("eps")
        result_path = results_dir / f"{path.stem}.result"
        rows.append((idx, float(eps), net, str(path), str(result_path)))
    rows.sort(key=lambda r: (r[0], r[1]))

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "eps", "net", "vnnlib", "result"])
        writer.writerows(rows)
    return manifest_path


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
        help="Generate specs for the first N test-split samples (indices 0..N-1). "
        "Overrides --indices. Use this to scale up incrementally toward the full test split.",
    )
    parser.add_argument(
        "--eps",
        default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1",
        help="Comma-separated L_inf eps values to sweep",
    )
    parser.add_argument("--k", type=int, default=8, help="top-k")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "vnnlib"))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Where run_batch.py will write results")
    parser.add_argument("--no-data-in-file", type=int, default=20000, help="Samples per original source file")
    parser.add_argument("--no-test-files", type=int, default=2, help="Number of trailing files reserved for test")
    args = parser.parse_args()

    if args.num_samples is not None:
        indices = list(range(args.num_samples))
    else:
        indices = [int(s) for s in args.indices.split(",")]
    eps_list = [float(s) for s in args.eps.split(",")]

    sess = ort.InferenceSession(args.net, providers=["CPUExecutionProvider"])
    n_out = sess.get_outputs()[0].shape[1]

    test_start, test_end = test_row_range(args.data, args.no_data_in_file, args.no_test_files)
    n_test = test_end - test_start
    if max(indices) >= n_test or min(indices) < 0:
        raise ValueError(f"--indices must be within the test split range [0, {n_test}), got range end {max(indices)}")

    # test 구간의 앞쪽부터 max(indices)+1 개 행만 읽는다 (test 구간 전체를 memmap으로 읽지 않음).
    X = load_row_range(args.data, test_start, test_start + max(indices) + 1)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(args.results_dir)

    n_written = 0
    n_skipped = 0
    for idx in indices:
        x0 = X[idx]
        y0, topk = clean_topk(sess, x0, args.k)
        print(f"idx={idx} (abs row {test_start + idx}) clean_top{args.k}={sorted(topk)}")
        for eps in eps_list:
            out_path = out_dir / f"mmimo_top{args.k}_idx{idx}_eps{eps:g}.vnnlib"
            if out_path.exists():
                n_skipped += 1
                continue
            write_vnnlib(out_path, x0, eps, topk, n_out=n_out)
            n_written += 1
            print(f"  wrote {out_path}")

    manifest_path = rebuild_manifest(out_dir, results_dir, net=args.net, k=args.k)
    print(f"wrote {n_written} new vnnlib files, skipped {n_skipped} existing ones")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
