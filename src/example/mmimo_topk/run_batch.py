"""
generate_vnnlib.py가 만든 manifest.csv(idx,eps,net,vnnlib,result)를 읽어서,
아직 결과가 없는 인스턴스마다 NeuralSAT(main.py)을 서브프로세스로 한 번씩 돌린다.

- main.py의 옵션 기본값(timeout=3600s 등)은 그대로 둔다 (오버라이드하지 않음).
- 이미 result 파일이 있는 인스턴스는 건너뛴다 -> 중간에 멈춰도 재실행하면 이어서 진행된다.
- 인스턴스 하나가 실패/예외가 나도 나머지는 계속 진행한다.
- --limit으로 이번 실행에서 새로 돌릴 인스턴스 개수를 제한할 수 있다 (전체 4만개 중
  일부만 우선 돌려보는 용도).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "vnnlib" / "manifest.csv"
DEFAULT_MAIN_PY = SCRIPT_DIR.parents[1] / "main.py"  # .../neuralsat/src/main.py


def read_manifest(manifest_path: pathlib.Path) -> list[dict]:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def run_one(main_py: pathlib.Path, net: str, vnnlib: str, result_path: pathlib.Path, log_path: pathlib.Path) -> int:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(main_py),
        "--net", net,
        "--spec", vnnlib,
        "--result_file", str(result_path),
        "--export_runtime",
        "--export_cex",
    ]
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv from generate_vnnlib.py")
    parser.add_argument("--main", default=str(DEFAULT_MAIN_PY), help="Path to neuralsat's src/main.py")
    parser.add_argument("--limit", type=int, default=None, help="Only attempt this many not-yet-run instances")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    main_py = pathlib.Path(args.main)
    rows = read_manifest(manifest_path)

    n_total = len(rows)
    n_skipped = 0
    n_attempted = 0
    n_failed = 0

    for row in rows:
        result_path = pathlib.Path(row["result"])
        if result_path.exists() and result_path.stat().st_size > 0:
            n_skipped += 1
            continue

        if args.limit is not None and n_attempted >= args.limit:
            break

        n_attempted += 1
        log_path = result_path.with_suffix(".log")
        tag = f"idx={row['idx']} eps={row['eps']}"

        if args.dry_run:
            print(f"[dry-run] would run {tag} -> {result_path}")
            continue

        start = time.time()
        returncode = run_one(main_py, row["net"], row["vnnlib"], result_path, log_path)
        elapsed = time.time() - start

        if returncode != 0 or not result_path.exists():
            n_failed += 1
            print(f"[FAILED] {tag} returncode={returncode} ({elapsed:.1f}s) see {log_path}")
            continue

        status_line = result_path.read_text().splitlines()[0] if result_path.stat().st_size > 0 else "<empty>"
        print(f"[{n_skipped + n_attempted}/{n_total}] {tag} -> {status_line} ({elapsed:.1f}s)")

    if args.dry_run:
        print(f"dry-run: {n_total} total, {n_skipped} already done, {n_attempted} would be run")
    else:
        print(
            f"done: {n_total} total, {n_skipped} already done, "
            f"{n_attempted - n_failed} newly completed, {n_failed} failed"
        )


if __name__ == "__main__":
    main()
