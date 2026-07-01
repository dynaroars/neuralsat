import re, subprocess, os, time

ROOT = '/home/roars/vnncomp26'
TODO = f'{ROOT}/todo.md'
PY = f'{ROOT}/venv/bin/python3'
TIMEOUT = 60  # short timeout per user request; full timeout run separately later
os.environ['GRB_LICENSE_FILE'] = os.path.expanduser('~/gurobi.lic')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

instances = []
with open(TODO) as f:
    for line in f:
        line = line.rstrip('\n')
        m = re.match(r'^- \[.\] ([a-zA-Z0-9_]+),(\S+\.onnx),(\S+\.vnnlib),', line)
        if m:
            instances.append((m.group(1), m.group(2), m.group(3)))
            continue
        if '\t' in line and line.count('\t') == 2:
            parts = line.split('\t')
            onnx_rel, vnnlib_rel = parts[0], parts[1]
            if onnx_rel.endswith('.onnx') and vnnlib_rel.endswith('.vnnlib'):
                # map old vnncomp2025_benchmarks paths -> vnncomp2026_benchmarks/.../1.0/...
                onnx_rel = re.sub(r'^vnncomp2025_benchmarks/benchmarks/(\w+)/onnx/', r'vnncomp2026_benchmarks/benchmarks/\1/1.0/onnx/', onnx_rel)
                vnnlib_rel = re.sub(r'^vnncomp2025_benchmarks/benchmarks/(\w+)/vnnlib/', r'vnncomp2026_benchmarks/benchmarks/\1/1.0/vnnlib/', vnnlib_rel)
                instances.append(('nn4sys', onnx_rel, vnnlib_rel))
                continue
        m3 = re.match(r'^([a-zA-Z0-9_]+),(\S+\.onnx),(\S+\.vnnlib),', line)
        if m3:
            instances.append((m3.group(1), m3.group(2), m3.group(3)))

print(f'Found {len(instances)} instances')

results = []
os.chdir(f'{ROOT}/neuralsat/src')
for bench, onnx_rel, vnnlib_rel in instances:
    onnx_path = f'{ROOT}/{onnx_rel}'
    vnnlib_path = f'{ROOT}/{vnnlib_rel}'
    if not os.path.exists(onnx_path) or not os.path.exists(vnnlib_path):
        results.append((bench, os.path.basename(vnnlib_rel), 'MISSING_FILE', 0))
        print(f'MISSING: {onnx_path} or {vnnlib_path}', flush=True)
        continue
    result_file = '/tmp/todo_run_result.txt'
    if os.path.exists(result_file):
        os.remove(result_file)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PY, 'main.py', '--net', onnx_path, '--spec', vnnlib_path,
             '--timeout', str(TIMEOUT), '--result_file', result_file],
            capture_output=True, text=True, timeout=TIMEOUT + 30,
        )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = 'proc_timeout'
    elapsed = time.time() - t0
    status = 'no_result_in_file'
    if os.path.exists(result_file):
        with open(result_file) as f:
            first = f.readline().strip()
            if first:
                status = first
    if exit_code != 0:
        status = f'error_exit_code_{exit_code}'
    results.append((bench, os.path.basename(vnnlib_rel), status, elapsed))
    print(f'{bench} | {os.path.basename(vnnlib_rel)} -> {status} ({elapsed:.1f}s)', flush=True)

print('\n=== SUMMARY ===')
for r in results:
    print(r)
