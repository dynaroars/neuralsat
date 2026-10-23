# NeuralSAT (minimal)

## Code style

- Functional programming only. Do not define any classes (no `class`,
  `@dataclass`, `NamedTuple` classes, `nn.Module` subclasses).
- Write module-level functions. Represent data with tensors, tuples, dicts and
  lists. Use closures instead of callable objects.
- Port reference code that uses classes (e.g.
  `~/Area/llmwiki/raw/codes/crown-minimal`) to functions.
- Clean code, PEP 8: lines at most 79 characters, small single-purpose
  functions, descriptive names, a short docstring per public function.
- Comments: use `'''` (not `"""`) for multi-line docstrings, and a plain
  `#` comment for single-line ones.
- DNN models are `nn.Sequential` only (Linear / ReLU). No ONNX support.

## Environment

- Python venv: `~/.local/share/venv`. Run Python and tests with
  `~/.local/share/venv/bin/python` / `~/.local/share/venv/bin/pytest`.

## Resource limits

- Machine: AMD Ryzen 9 5900X (12 cores, 24 logical CPUs), ASUS TUF GAMING
  B550-PLUS. At most 23 threads in total (one logical CPU stays free), and at
  most 2 processes in parallel (no `pytest -n`/xdist, no `make -j`, at most 2
  background jobs or subagents).
- Set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to at most 23
  for every run (the makefiles do) and use `torch.set_num_threads(THREADS)`
  with `THREADS` from `writeup/exp/config.py` in scripts.
- Keep experiments small (small networks, few samples); avoid long loops
  over many seeds.

### History of the CPU fault (read this after a random shutdown or freeze)

- 2026-10-23: heavy parallel load crashed the machine. A thread ramp
  (`python -m writeup.exp.threads`, CSV `writeup/data/threads.csv`) found
  threads 1 to 10 clean and a corrected ECC error in the L1 data cache of
  logical CPU 14 (MC0, load/store unit; sibling thread of CPU 2) at 11
  threads: `mce: [Hardware Error] ... cache level: L1, tx: DATA`. The `Over`
  flag showed that earlier errors had been lost.
- After BIOS changes and a reboot, the ramp ran 10 to 24 threads (all logical
  CPUs) with no machine-check message, and the limit was raised to 23. One
  clean pass of 20 s per level is not proof, because the error is
  intermittent.
- If the machine shuts down, freezes or reboots at random, assume the CPU
  cache-level error is back before debugging the project code:
  1. Check `journalctl -k -b -1 | grep -iE "hardware error|machine check"`
     (the previous boot) and `journalctl -k -b 0` for new lines, and look
     for the CPU number and the cache level (L1/L2).
  2. Lower the limit to 10 threads (the last count that was clean before the
     BIOS change): `THREADS` in `writeup/exp/config.py` and the `*_NUM_THREADS`
     lines in `makefile` and `writeup/makefile`.
  3. Rerun the ramp (`python -m writeup.exp.threads --start 10`; it stops at
     the first machine-check message) to find the new safe level.
  4. In the BIOS (Delete/F2 at power-on) set PBO off and Curve Optimizer to 0,
     or a positive offset on core 2, or update the BIOS; or take CPUs 2 and 14
     offline / avoid them with `taskset -c`; if it persists at stock
     settings the CPU is faulty (warranty).
