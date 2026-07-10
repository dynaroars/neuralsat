"""
manifest.csv(idx,eps,net,vnnlib,result)와 run_batch.py가 만든 result 파일들을
모아서 summary.csv(idx,eps,status,runtime)와 idx별 certified-robust 반경을
robustness_summary.csv로 정리한다.

robust radius 정의: eps를 오름차순으로 볼 때, 가장 작은 eps부터 연속으로
unsat인 구간의 마지막 eps (= 그보다 작은 eps는 전부 unsat으로 확인된 반경).
중간에 unsat이 아닌 결과가 나온 뒤 더 큰 eps에서 다시 unsat이 나오면(비단조),
anomaly로 표시한다 (L_inf eps-ball이 커질수록 sat/unknown 쪽으로 가는게
자연스러운데 그 가정이 깨진 경우).
"""

from __future__ import annotations

import argparse
import csv
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "vnnlib" / "manifest.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "summary.csv"
DEFAULT_ROBUSTNESS_SUMMARY = SCRIPT_DIR / "robustness_summary.csv"


def read_result(result_path: pathlib.Path) -> tuple[str, str]:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return "not_run", ""
    first_line = result_path.read_text().splitlines()[0]
    parts = first_line.split(",", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--robustness-out", default=str(DEFAULT_ROBUSTNESS_SUMMARY))
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        status, runtime = read_result(pathlib.Path(row["result"]))
        records.append({"idx": int(row["idx"]), "eps": float(row["eps"]), "status": status, "runtime": runtime})

    with open(args.summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "eps", "status", "runtime"])
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["idx"], r["eps"])):
            writer.writerow(r)

    by_idx: dict[int, list[dict]] = {}
    for r in records:
        by_idx.setdefault(r["idx"], []).append(r)

    robustness_rows = []
    status_counts: dict[str, int] = {}
    for idx, recs in sorted(by_idx.items()):
        recs.sort(key=lambda r: r["eps"])
        robust_radius = None
        break_eps = None
        break_status = None
        anomaly = False
        broke = False
        for r in recs:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            if not broke:
                if r["status"] == "unsat":
                    robust_radius = r["eps"]
                else:
                    broke = True
                    break_eps = r["eps"]
                    break_status = r["status"]
            elif r["status"] == "unsat":
                anomaly = True

        robustness_rows.append(
            {
                "idx": idx,
                "robust_radius_eps": robust_radius if robust_radius is not None else "",
                "first_non_unsat_eps": break_eps if break_eps is not None else "",
                "first_non_unsat_status": break_status if break_status is not None else "",
                "anomaly_nonmonotonic": anomaly,
            }
        )

    with open(args.robustness_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "robust_radius_eps", "first_non_unsat_eps", "first_non_unsat_status", "anomaly_nonmonotonic"],
        )
        writer.writeheader()
        writer.writerows(robustness_rows)

    print(f"summary: {len(records)} instances across {len(by_idx)} data points")
    print(f"status counts: {status_counts}")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.robustness_out}")


if __name__ == "__main__":
    main()
