"""
generate_vnnlib.py가 만든 manifest.csv(idx,eps,net,data,k,abs_row,result)를 읽어서,
아직 결과가 없는 인스턴스마다 NeuralSAT(main.py)을 서브프로세스로 한 번씩 돌린다.

VNNLIB 텍스트는 디스크에 영구 저장하지 않는다: 인스턴스를 실행하기 직전에
(x0, clean_topk)로부터 임시 파일에 즉석 생성하고, main.py 실행이 끝나면(성공/실패
무관) 바로 삭제한다. 인스턴스당 ~26KB인 VNNLIB 텍스트가 최종 목표(28만 인스턴스)
기준 수GB + 파일 수십만 개로 쌓이는 것을 막기 위함이다. 영구적으로 남는 것은
manifest.csv(가벼운 idx/eps 목록)와 실제 검증 결과(results/*.result, *.log)뿐이다.

- main.py의 옵션 기본값(timeout=3600s 등)은 그대로 둔다 (오버라이드하지 않음).
- 이미 result 파일이 있는 인스턴스는 건너뛴다 -> 중간에 멈춰도 재실행하면 이어서 진행된다.
- 인스턴스 하나가 실패/예외가 나도 나머지는 계속 진행한다.
- --limit으로 이번 실행에서 새로 돌릴 idx(데이터 샘플) 개수를 제한할 수 있다
  (전체 4만개 중 일부만 우선 돌려보는 용도. 병렬화 이후로는 인스턴스가 아니라
  idx 단위로 센다).

## eps 조기 종료(early-stop)

eps ball은 커질수록 이전 ball을 완전히 포함하는 상위집합이다. 그래서 어떤 idx에서
eps=e에 대해 이미 "sat"(top-k를 바꾸는 반례 발견)이 확인됐다면, 그 반례는
eps=e' > e 인 더 큰 ball에도 그대로 속하므로 eps=e'도 수학적으로 반드시 sat이다
(NeuralSAT을 다시 돌려서 확인할 필요가 없다). 이 성질을 이용해 idx별로 eps
오름차순으로 보다가 실제/기존 결과가 "sat"으로 확정되는 순간부터는 그 idx의
남은(더 큰) eps는 NeuralSAT을 실행하지 않고 바로 result에 "sat,0"을 기록한다
(log 파일에 "[implied]"로 왜 건너뛰었는지 남긴다).

"unknown"(timeout 등)에서는 이 논리가 성립하지 않으므로(더 큰 eps가 sat인지
unsat인지 unknown 결과만으로는 알 수 없음) 조기 종료하지 않고 그대로 실행한다.
"unsat"도 당연히 다음 eps를 계속 실행한다.

## 병렬화 (--workers)

eps 조기 종료가 idx 하나 안에서 "작은 eps부터 순서대로" 봐야 성립하므로, 병렬화
단위는 인스턴스(idx, eps) 한 줄이 아니라 **idx(데이터 샘플) 하나**다. idx마다
자신의 eps 스윕(조기 종료 포함)을 처음부터 끝까지 순차로 처리하는 작업을
`ProcessPoolExecutor`로 여러 idx에 대해 동시에 돌린다. 이렇게 하면 idx 간
완전히 독립적인 작업을 CPU 코어 수만큼 병렬로 처리하면서도, idx 안에서의
조기 종료 이점은 그대로 유지된다.

여러 NeuralSAT 서브프로세스가 동시에 도는 동안 각 프로세스가 numpy/torch 기본
스레드 풀로 모든 코어를 잡으려 들면 오히려 코어 경쟁으로 느려질 수 있어서,
`--workers`에 맞춰 OMP_NUM_THREADS 등을 프로세스당 코어수/workers로 낮춰서
넘긴다 (Gurobi 자체 스레드는 대부분 NeuralSAT 코드에서 이미 Threads=1로
고정하고 있으나, 일부 경로는 기본값을 쓸 수 있어 완전히 통제되진 않는다 -
`--workers`를 올렸는데 체감 속도가 기대만큼 안 나오면 낮춰서 다시 시도해볼 것).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import onnxruntime as ort
import psutil

from generate_vnnlib import build_vnnlib_text, clean_topk
from pickle_memmap import load_row_range

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "vnnlib" / "manifest.csv"
DEFAULT_MAIN_PY = SCRIPT_DIR.parents[1] / "main.py"  # .../neuralsat/src/main.py

# 워커 프로세스 하나가 여러 idx를 처리할 수 있으므로, 프로세스 안에서 onnxruntime
# 세션을 net 경로별로 캐싱해서 매 idx마다 새로 로드하지 않게 한다.
_SESSION_CACHE: dict[str, ort.InferenceSession] = {}


def read_manifest(manifest_path: pathlib.Path) -> list[dict]:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def get_session(net: str) -> ort.InferenceSession:
    sess = _SESSION_CACHE.get(net)
    if sess is None:
        sess = ort.InferenceSession(net, providers=["CPUExecutionProvider"])
        _SESSION_CACHE[net] = sess
    return sess


def make_vnnlib_file(sess: ort.InferenceSession, data: str, abs_row: int, eps: float, k: int, tmp_dir: pathlib.Path) -> pathlib.Path:
    x0 = load_row_range(data, abs_row, abs_row + 1)[0]
    n_out = sess.get_outputs()[0].shape[1]
    _, topk = clean_topk(sess, x0, k)
    text = build_vnnlib_text(x0, eps, topk, n_out)

    fd, tmp_path = tempfile.mkstemp(suffix=".vnnlib", dir=tmp_dir)
    with open(fd, "w") as f:
        f.write(text)
    return pathlib.Path(tmp_path)


def read_status(result_path: pathlib.Path) -> str | None:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return None
    return result_path.read_text().splitlines()[0].split(",", 1)[0]


def write_implied_sat(result_path: pathlib.Path, log_path: pathlib.Path, tag: str) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("sat,0\n")
    log_path.write_text(
        f"[implied] {tag}: skipped NeuralSAT run. A counterexample already found at a smaller eps "
        "for this idx remains valid at this larger eps (eps-ball is a superset), so this is sat "
        "by construction, not independently verified.\n"
    )


def run_one(main_py: pathlib.Path, net: str, spec: pathlib.Path, result_path: pathlib.Path, log_path: pathlib.Path, extra_env: dict[str, str]) -> int:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(main_py),
        "--net", net,
        "--spec", str(spec),
        "--result_file", str(result_path),
        "--export_runtime",
        "--export_cex",
    ]
    env = os.environ.copy()
    env.update(extra_env)
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def process_idx_group(idx: int, rows: list[dict], main_py: str, tmp_dir: str, thread_env: dict[str, str]) -> dict:
    """idx 하나의 eps 오름차순 시퀀스를 조기 종료 포함해서 순차로 처리한다.

    워커 프로세스 안에서 실행되므로 print를 직접 하지 않고, 메인 프로세스가 순서대로
    출력할 수 있도록 (라인 종류, 텍스트) 목록을 모아서 반환한다.
    """
    tmp_dir_p = pathlib.Path(tmp_dir)
    main_py_p = pathlib.Path(main_py)
    lines: list[tuple[str, str]] = []
    n_skipped = n_attempted = n_failed = n_implied = 0
    sat_seen = False

    for row in rows:
        result_path = pathlib.Path(row["result"])
        log_path = result_path.with_suffix(".log")
        tag = f"idx={row['idx']} eps={row['eps']}"

        existing_status = read_status(result_path)
        if existing_status is not None:
            if existing_status == "sat":
                sat_seen = True
            n_skipped += 1
            continue

        if sat_seen:
            write_implied_sat(result_path, log_path, tag)
            n_implied += 1
            lines.append(("ok", f"{tag} -> sat (implied, 0.0s)"))
            continue

        n_attempted += 1
        start = time.time()
        net = row["net"]
        sess = get_session(net)
        spec_path = make_vnnlib_file(sess, row["data"], int(row["abs_row"]), float(row["eps"]), int(row["k"]), tmp_dir_p)
        try:
            returncode = run_one(main_py_p, net, spec_path, result_path, log_path, thread_env)
        finally:
            spec_path.unlink(missing_ok=True)
        elapsed = time.time() - start

        if returncode != 0 or not result_path.exists():
            n_failed += 1
            lines.append(("failed", f"[FAILED] {tag} returncode={returncode} ({elapsed:.1f}s) see {log_path}"))
            continue

        status_line = result_path.read_text().splitlines()[0] if result_path.stat().st_size > 0 else "<empty>"
        if status_line.split(",", 1)[0] == "sat":
            sat_seen = True
        lines.append(("ok", f"{tag} -> {status_line} ({elapsed:.1f}s)"))

    return {
        "idx": idx,
        "lines": lines,
        "n_skipped": n_skipped,
        "n_attempted": n_attempted,
        "n_failed": n_failed,
        "n_implied": n_implied,
    }


def group_by_idx(rows: list[dict]) -> list[tuple[int, list[dict]]]:
    rows = sorted(rows, key=lambda r: (int(r["idx"]), float(r["eps"])))
    return [(idx, list(grp)) for idx, grp in itertools.groupby(rows, key=lambda r: int(r["idx"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.csv from generate_vnnlib.py")
    parser.add_argument("--main", default=str(DEFAULT_MAIN_PY), help="Path to neuralsat's src/main.py")
    parser.add_argument("--limit", type=int, default=None, help="Only attempt this many not-yet-fully-done idx groups")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of idx (data sample) groups to verify in parallel. "
        "Default: physical_core_count - 1 (leaves one core free; hyperthreaded logical cores "
        "don't help much for Gurobi/LP-heavy verification, so we size off physical cores). "
        "Use 1 for sequential (old) behavior.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    main_py = pathlib.Path(args.main)
    rows = read_manifest(manifest_path)
    groups = group_by_idx(rows)

    n_total = len(rows)
    n_skipped = 0
    n_attempted = 0
    n_failed = 0
    n_implied = 0

    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    workers = args.workers if args.workers is not None else max(1, physical_cores - 1)

    tmp_dir = manifest_path.parent / "_tmp_vnnlib"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 이미 전부 끝난 idx는 건너뛰고, 아직 할 일이 남은 idx만 --limit 만큼 골라낸다.
    pending_groups: list[tuple[int, list[dict]]] = []
    for idx, grp_rows in groups:
        if all(read_status(pathlib.Path(r["result"])) is not None for r in grp_rows):
            n_skipped += len(grp_rows)
            continue
        if args.limit is not None and len(pending_groups) >= args.limit:
            break
        pending_groups.append((idx, grp_rows))

    if args.dry_run:
        for idx, grp_rows in pending_groups:
            sat_seen = False
            for row in grp_rows:
                result_path = pathlib.Path(row["result"])
                tag = f"idx={row['idx']} eps={row['eps']}"
                status = read_status(result_path)
                if status is not None:
                    if status == "sat":
                        sat_seen = True
                    continue
                if sat_seen:
                    print(f"[dry-run] would mark {tag} as sat (implied, no run)")
                else:
                    print(f"[dry-run] would run {tag} -> {result_path}")
        print(f"dry-run: {n_total} total, {n_skipped} already done, {len(pending_groups)} idx groups would be run (workers={workers})")
        return

    thread_env: dict[str, str] = {}
    if workers > 1:
        threads_per_worker = max(1, physical_cores // workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            thread_env[var] = str(threads_per_worker)

    counter = n_skipped

    def consume(result: dict) -> None:
        nonlocal counter, n_attempted, n_failed, n_implied
        n_attempted += result["n_attempted"]
        n_failed += result["n_failed"]
        n_implied += result["n_implied"]
        for kind, text in result["lines"]:
            if kind == "ok":
                counter += 1
                print(f"[{counter}/{n_total}] {text}")
            else:
                print(text)

    if workers <= 1:
        for idx, grp_rows in pending_groups:
            consume(process_idx_group(idx, grp_rows, str(main_py), str(tmp_dir), thread_env))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_idx_group, idx, grp_rows, str(main_py), str(tmp_dir), thread_env) for idx, grp_rows in pending_groups]
            for fut in as_completed(futures):
                consume(fut.result())

    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty or in use; leave it

    print(
        f"done: {n_total} total, {n_skipped} already done, "
        f"{n_attempted - n_failed} newly completed, {n_implied} implied (skipped), {n_failed} failed "
        f"(workers={workers})"
    )


if __name__ == "__main__":
    main()
