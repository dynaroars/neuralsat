# mMIMO top-k 안테나 선택 강건성 검증 파이프라인

mMIMO 안테나 선택 네트워크(256차원 입력 -> 16차원 출력, top-k 선택)의 로컬
강건성(local robustness)을 NeuralSAT으로 검증하는 파이프라인. 원본 pickle
데이터셋에서 test 구간만 떼어내 (데이터 idx, eps) 조합마다 VNNLIB 스펙을 만들고,
NeuralSAT을 인스턴스별로 돌린 뒤 결과를 모아 idx별 certified-robust 반경을
정리하는 것까지 4단계로 구성된다.

## 파이프라인 개요

```
data_split.py / pickle_memmap.py   (test 구간 판별용 유틸, 직접 실행 X)
        |
        v
generate_vnnlib.py   --num-samples N --eps ...   ->  vnnlib/*.vnnlib + vnnlib/manifest.csv
        |
        v
run_batch.py          --manifest vnnlib/manifest.csv   ->  results/*.result (+ *.log)
        |
        v
summarize_results.py  --manifest vnnlib/manifest.csv   ->  summary.csv, robustness_summary.csv
```

각 단계는 독립 실행 가능하고, 모두 **manifest.csv**(idx, eps, net, data, k,
abs_row, result) 하나를 공통 인터페이스로 사용한다. 이미 결과가 있는 인스턴스는
다시 돌리지 않으므로 중간에 멈춰도 이어서 진행할 수 있다(resume).

**VNNLIB 텍스트 파일은 디스크에 영구 저장하지 않는다.** 인스턴스 하나당
~26KB인데, 최종 목표(test 구간 전체 4만개 x eps 23개 = 92만 인스턴스)까지
가면 수GB + 파일 수십만 개가 쌓여 공간과 파일시스템 성능을 심하게 낭비하기 때문이다.
대신 `run_batch.py`가 NeuralSAT을 호출하기 직전에 (x0, clean_topk)로부터 임시
파일로 즉석 생성하고, 실행이 끝나면(성공/실패 무관) 바로 삭제한다. 영구적으로
남는 것은 manifest.csv(가벼운 idx/eps 목록)와 실제 검증 결과
(results/*.result, *.log)뿐이다.

## 1. 데이터 분할 (`data_split.py`, `pickle_memmap.py`)

원본 pickle(`mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D-003.pickle`, 약 3.2GB,
`(1,600,000, 256)` float64 = 80개 파일 × 20,000샘플)은 전체를 메모리에 올리지
않고 `pickle_memmap.py`의 memmap 유틸로 필요한 행 구간만 읽는다.

- `pickle_memmap.peek_ndarray_info` : pickle 헤더만 읽어 shape/dtype/데이터
  offset을 알아낸다 (raw 데이터는 안 읽음).
- `pickle_memmap.load_row_range(path, start, end)` : `[start, end)` 행만
  memmap으로 읽어온다.
- `data_split.test_row_range(path, no_dataInFile, no_test_files)` : 전체
  행 수만으로 test 구간의 절대 행 범위를 계산한다. 기본값(`no_dataInFile=20000,
  no_test_files=2`) 기준 test 구간은 뒤에서 2개 파일분, 즉 절대 행
  `1,560,000 ~ 1,600,000` (40,000개)이다. **train 구간(앞 78개 파일)은
  검증 파이프라인에서 절대 사용하지 않는다.**

이 두 모듈은 `generate_vnnlib.py`가 import해서 쓰며, 직접 실행할 일은 없다.

## 2. VNNLIB 인스턴스 목록 생성 (`generate_vnnlib.py`)

test 구간 내 상대 인덱스(0 = test 구간 첫 샘플)를 기준으로, 각 (idx, eps) 조합을
manifest.csv 행으로 등록한다. 실제 VNNLIB 텍스트("clean top-k 선택이 절대 안
바뀐다"의 부정을 표현하는 스펙)는 이 단계에서 만들지 않고, `run_batch.py`가
실행 직전에 즉석 생성한다(아래 3단계 참고). NeuralSAT이 `unsat`을 내면 그
반경 안에서 top-k 선택이 절대 바뀌지 않음이 증명된 것이고(certified robust),
`sat`이면 반례가 존재하는 것이다.

```bash
# test 구간 첫 100개 샘플 x 기본 eps 리스트(1e-6..1) 를 manifest에 등록
python generate_vnnlib.py --num-samples 100

# 이미 등록된 조합은 건너뛰고, 새로 200개까지 확장
python generate_vnnlib.py --num-samples 200
```

주요 옵션:
- `--num-samples N` : `--idx-start`부터 N개 샘플(기본 idx 0..N-1)에 대해 등록.
  **최종 목표는 test 구간 전체(40,000개)이지만, 지금은 이 옵션으로 원하는
  개수만큼만 점진적으로 늘려간다.** (`--indices`로 특정 idx만 지정하는 것도
  여전히 가능하지만 `--num-samples`가 주어지면 무시된다.)
- `--idx-start` : `--num-samples`와 같이 써서 시작 idx를 지정 (기본 0). 여러
  대의 컴퓨터가 test 구간을 나눠서 처리할 때 사용 (예: 컴퓨터 A는
  `--idx-start 0 --num-samples 20000`, 컴퓨터 B는
  `--idx-start 20000 --num-samples 20000`). **idx는 0-indexed**라서
  "20001번째 샘플"은 `idx=20000`이다.
- `--eps` : 기본 `1e-6,1e-5,1e-4,2e-4,...,9e-4,1e-3,2e-3,...,9e-3,1e-2,1e-1,1`
  (23개, 반경별 sat 비율이 급격히 갈리는 1e-4~1e-3 구간을 촘촘하게 잡은 고정 리스트).
- `--k` : top-k, 기본 8.
- `--out-dir` : manifest.csv 저장 위치, 기본 `vnnlib/`.
- `--results-dir` : manifest.csv에 적힐 result 파일 위치(실제 실행은
  run_batch.py가 함), 기본 `results/`.

실행할 때마다 기존 `manifest.csv`를 읽어 이미 등록된 (idx, eps) 행(및 그
result 여부)은 그대로 유지하면서 새로 요청된 조합만 추가한다. 그래서 여러 번
나눠서 `--num-samples`를 늘려가며 실행해도 안전하고, 디스크에는 매번 가벼운
manifest.csv 한 장만 남는다.

## 3. NeuralSAT 배치 실행 (`run_batch.py`)

`manifest.csv`를 읽어서, 아직 result 파일이 없는 인스턴스마다 (x0, clean_topk)로
VNNLIB 텍스트를 임시 파일로 즉석 생성하고 `src/main.py`를 서브프로세스로 한 번씩
호출한 뒤, 실행이 끝나면(성공/실패 무관) 그 임시 파일을 바로 삭제한다. `main.py`의
옵션 기본값(timeout=3600s 등)은 그대로 사용하고 오버라이드하지 않는다.

eps 조기 종료가 idx 하나 안에서 작은 eps부터 순서대로 봐야 성립하므로, 병렬화
단위는 **idx(데이터 샘플) 하나**다. idx마다 자신의 eps 스윕(조기 종료 포함)을
순차로 처리하는 작업을 `--workers`개의 프로세스로 동시에 돌린다 (idx끼리는
서로 완전히 독립적이라 병렬화하기 좋음). 기본값은 **물리 코어 수 - 1**(코어
하나는 비워둠, `psutil.cpu_count(logical=False)` 기준 — Gurobi/LP 위주
작업이라 하이퍼스레딩으로 늘어난 논리 코어는 크게 도움이 안 돼서 물리 코어로
계산)이고, `--workers 1`을 주면 예전과 같은 순차 실행이 된다.

```bash
# 아직 안 돌린 인스턴스 전부 실행 (기본 workers = 물리 코어 수 - 1)
python run_batch.py

# 워커 수를 직접 지정 (6코어 머신 기준 예시)
python run_batch.py --workers 5

# 이번엔 idx 5개만 새로 실행 (파일럿/점검용, --limit은 이제 idx 단위)
python run_batch.py --limit 5

# 실제로 돌리지 않고 무엇을 실행할지만 확인
python run_batch.py --dry-run
```

4개 idx(28개 인스턴스, 그중 15개만 실제 실행되고 13개는 조기 종료로 스킵)를
`--workers 3`으로 돌려본 결과, 실제 걸린 시간은 약 59초였다(순차로 다 돌렸을 때
예상되는 약 197초 대비 약 3.3배). 여러 NeuralSAT 서브프로세스가 동시에 돌 때
numpy/torch 스레드가 코어를 서로 잡아먹지 않도록 `OMP_NUM_THREADS` 등을
`cpu_count // workers`로 낮춰서 넘기지만, Gurobi 내부 스레드는 일부 경로에서
완전히 통제되지 않으므로 `--workers`를 올렸는데 체감 속도가 기대만큼 안 나오면
낮춰서 다시 시도해볼 것.

- 이미 result 파일이 있는 (idx, eps)는 건너뛴다 -> 중단 후 재실행하면
  이어서 진행된다.
- 실행이 끝나면 `this session`(이번 실행 소요 시간)과 `cumulative total`(이
  manifest에 대해 지금까지 실행한 모든 세션의 합, 세션 사이 공백은 안 셈)을
  같이 출력한다. `vnnlib/.run_time_state.json`에 누적값을 저장해두므로,
  `--num-samples`를 늘려가며 여러 번 나눠 돌려도 실제 가동 시간의 총합을
  정확히 알 수 있다.
- 인스턴스 하나가 실패해도(리턴코드 != 0) 나머지는 계속 진행하고, 실패한
  인스턴스는 `[FAILED]`로 표시된다 (원인은 `results/*.log` 참고).
- 각 인스턴스 실행 시 `--export_runtime --export_cex`를 추가로 넘겨서, 결과
  파일 첫 줄에 `status,runtime`을, sat인 경우 둘째 줄에 counterexample을 남긴다.

## 4. 결과 집계 (`summarize_results.py`)

```bash
python summarize_results.py
```

- `manifest.csv` + `results/*.result`를 모아 `summary.csv`(idx, eps, status,
  runtime)를 만든다. 아직 안 돌린 인스턴스는 status가 `not_run`으로 표시된다.
- idx별로 eps 오름차순으로 봤을 때, 가장 작은 eps부터 연속으로 `unsat`인
  구간의 마지막 eps를 **certified-robust 반경**으로 잡아 `robustness_summary.csv`
  에 정리한다 (`robust_radius_eps`, 그 다음 첫 non-unsat eps/status,
  `anomaly_nonmonotonic` — eps가 커질수록 sat/unknown 쪽으로 가는 게 자연스러운데
  중간에 끊겼다가 더 큰 eps에서 다시 unsat이 나오면 표시).
- eps별로 전체 idx 대비 **unsat** 비율을 막대그래프로 그려 `eps_unsat_ratio.png`에
  저장한다(`--chart-out`으로 경로 변경 가능). 새 의존성 없이 이미 있는
  Pillow만으로 그린 PNG 이미지다.

## 알려진 이슈 / 진행 상황

- `neuralsat` conda 환경에 `requirements.txt` 전체 설치 완료, idx=0(eps 7개)
  기준으로 생성 -> run_batch(실제 `main.py` 서브프로세스 실행까지) -> summarize
  전체 파이프라인이 end-to-end로 정상 동작함을 확인함(1번째 데이터 기준
  robust radius = eps 1e-5).
- VNNLIB 텍스트를 `.vnnlib` 파일로 영구 저장하던 초기 구조는 인스턴스당
  ~26KB라 최종 규모(28만 인스턴스)에서 수GB + 파일 수십만 개가 쌓이는 문제가
  있어, `run_batch.py`가 실행 직전에 임시 파일로 즉석 생성 후 즉시 삭제하는
  방식으로 변경함. 디스크에는 manifest.csv와 results/*.result, *.log만 남는다.
- 최종적으로는 test 구간 40,000개 전부를 대상으로 하되, 지금은
  `--idx-start`/`--num-samples`로 여러 컴퓨터가 idx 구간을 나눠서 진행 중이다
  (예: 컴퓨터 A는 idx 0~19999, 컴퓨터 B는 idx 20000~39999). eps는 23개로
  늘렸으므로 40,000 x 23 = 92만 인스턴스 규모다. 인스턴스당 실행 시간에 따라
  매우 오래 걸릴 수 있으므로, 파일럿 결과를 보고 timeout/샘플 수/병렬화 여부를
  다시 판단해야 한다.

1. 인스턴스 목록(manifest) 100개 생성 
python generate_vnnlib.py --idx-start 20000 --num-samples 20000

2. neuralsat 배치 실행
python run_batch.py

3. 결과 집계
python summarize_results.py