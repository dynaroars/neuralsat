import os
import sys
import uuid
import json
import time
import shutil
import subprocess
import threading
import tempfile
import re
import signal
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

NEURALSAT_ROOT = Path(os.environ.get("NEURALSAT_ROOT", Path(__file__).resolve().parent.parent))
NEURALSAT_MAIN = NEURALSAT_ROOT / "src" / "main.py"
EXAMPLE_ONNX_DIR = NEURALSAT_ROOT / "src" / "example" / "onnx"
EXAMPLE_VNNLIB_DIR = NEURALSAT_ROOT / "src" / "example" / "vnnlib"
CLASSIC_DIR = Path(__file__).resolve().parent / "frontend-classic"

UPLOAD_DIR = Path(tempfile.gettempdir()) / f"neuralsat_uploads_{os.getuid()}"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_TIMEOUT = 300
MAX_TIMEOUT = 3600

# Settings files must live here; clients may only reference a bare filename inside it.
SETTINGS_DIR = NEURALSAT_ROOT / "src" / "example" / "settings"

SHAPE_TOKEN_RE = re.compile(r'^\d+$')


def _parse_shape(raw):
    """Validate a whitespace-separated list of positive integers.

    Returns (tokens, error). Rejecting anything else prevents extra argv
    tokens (e.g. another `--flag`) from being smuggled into the solver
    subprocess's command line via this field.
    """
    if not raw or not raw.strip():
        return None, None
    tokens = raw.split()
    if not all(SHAPE_TOKEN_RE.match(t) for t in tokens):
        return None, "must be whitespace-separated positive integers"
    return tokens, None


def _resolve_setting_file(raw):
    """Resolve a client-supplied settings filename against SETTINGS_DIR.

    Only a bare filename is accepted; anything containing a path separator,
    or that resolves outside SETTINGS_DIR, is rejected. This prevents the
    field from being used to read arbitrary files on the server.
    """
    if not raw or not raw.strip():
        return None, None
    name = raw.strip()
    if "/" in name or "\\" in name or name in (".", ".."):
        return None, "must be a bare filename, not a path"

    base = SETTINGS_DIR.resolve()
    candidate = (base / name).resolve()
    if candidate != base and base not in candidate.parents:
        return None, "must be a bare filename, not a path"
    if not candidate.is_file():
        return None, f"settings file not found: {name}"
    return candidate, None

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

jobs: dict[str, dict] = {}
job_lock = threading.Lock()
run_lock = threading.Lock()

def _cleanup_old_jobs(max_age_s: int = 3600):

    now = time.time()
    with job_lock:
        expired = [jid for jid, j in jobs.items() if now - j["created"] > max_age_s]
        for jid in expired:
            job = jobs.pop(jid)
            job_dir = Path(job.get("job_dir", ""))
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)

def _get_onnx_info(net_path):
    try:
        import onnx
        model = onnx.load(str(net_path))
        info = {
            "layers": [],
            "inputs": [],
            "outputs": [],
            "num_parameters": 0,
            "num_layers": 0,
            "num_neurons": 0
        }

        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else "dynamic")
            info["inputs"].append({"name": inp.name, "shape": shape})

        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else "dynamic")
            info["outputs"].append({"name": out.name, "shape": shape})

        for node in model.graph.node:
            info["layers"].append({
                "op": node.op_type,
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output)
            })

        # Count parameters
        num_parameters = 0
        for init in model.graph.initializer:
            p = 1
            for d in init.dims:
                p *= d
            num_parameters += p
        info["num_parameters"] = num_parameters

        # Count layers
        info["num_layers"] = len(model.graph.node)

        # Count neurons / nodes
        num_neurons = 0
        try:
            from onnx import shape_inference
            inferred = shape_inference.infer_shapes(model)
            value_info = {vi.name: vi for vi in inferred.graph.value_info}
            for out in inferred.graph.output:
                value_info[out.name] = out
            for node in inferred.graph.node:
                if not node.output:
                    continue
                out_name = node.output[0]
                if out_name in value_info:
                    vi = value_info[out_name]
                    shape = []
                    for dim in vi.type.tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(None)
                    if len(shape) > 0:
                        prod = 1
                        has_dims = False
                        for d in shape[1:]:
                            if d is not None and d > 0:
                                prod *= d
                                has_dims = True
                        if has_dims:
                            num_neurons += prod
                        elif shape[0] is not None and shape[0] > 0:
                            num_neurons += shape[0]
        except Exception:
            pass

        # Fallback to sum of bias/1D initializers if no neurons counted
        if num_neurons == 0:
            for init in model.graph.initializer:
                if len(init.dims) == 1:
                    num_neurons += init.dims[0]
        
        info["num_neurons"] = num_neurons
        return info
    except Exception as e:
        return {
            "error": f"Failed to parse ONNX: {str(e)}", 
            "layers": [], 
            "inputs": [], 
            "outputs": [],
            "num_parameters": 0,
            "num_layers": 0,
            "num_neurons": 0
        }

def _get_vnnlib_info(spec_path):
    try:
        content = Path(spec_path).read_text()

        vars_decl = re.findall(r'\(declare-const\s+([\w\d_]+)\s+Real\)', content)
        if not vars_decl:
            vars_decl = re.findall(r'\(declare-const\s+([\w\d_]+)\s+', content)

        bounds = {v: {"lower": None, "upper": None} for v in vars_decl}

        for line in content.splitlines():
            line = line.strip()
            if not line.startswith("(assert"):
                continue

            m1 = re.search(r'\(<=\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+([\w\d_]+)\)', line)
            if m1:
                val = float(m1.group(1))
                var = m1.group(2)
                if var in bounds:
                    bounds[var]["lower"] = val

            m2 = re.search(r'\(<=\s+([\w\d_]+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)', line)
            if m2:
                var = m2.group(1)
                val = float(m2.group(2))
                if var in bounds:
                    bounds[var]["upper"] = val

            m3 = re.search(r'\(>=\s+([\w\d_]+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)', line)
            if m3:
                var = m3.group(1)
                val = float(m3.group(2))
                if var in bounds:
                    bounds[var]["lower"] = val

            m4 = re.search(r'\(>=\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+([\w\d_]+)\)', line)
            if m4:
                val = float(m4.group(1))
                var = m4.group(2)
                if var in bounds:
                    bounds[var]["upper"] = val

        # Extract balanced assertions to find output properties
        assertions = []
        pos = 0
        while True:
            idx = content.find("(assert", pos)
            if idx == -1:
                break
            # trace matching parentheses
            paren_count = 0
            end_idx = idx
            for i in range(idx, len(content)):
                if content[i] == '(':
                    paren_count += 1
                elif content[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_idx = i + 1
                        break
            if paren_count == 0:
                assertions.append(content[idx:end_idx].strip())
                pos = end_idx
            else:
                pos = idx + 7 # skip past '(assert'

        # Filter output constraints
        output_vars = [v for v in vars_decl if v.lower().startswith('y')]
        output_constraints = []
        for assert_str in assertions:
            if any(v in assert_str for v in output_vars):
                output_constraints.append(assert_str)

        return {
            "variables": vars_decl,
            "bounds": bounds,
            "output_constraints": output_constraints,
            "raw_content": content
        }
    except Exception as e:
        return {
            "error": f"Failed to parse VNNLib: {str(e)}",
            "variables": [],
            "bounds": {},
            "output_constraints": [],
            "raw_content": ""
        }

def _run_verification(job_id: str):

    with job_lock:
        job = jobs[job_id]
        job["status"] = "running"
        job["started"] = time.time()

    net_path = job["net_path"]
    spec_path = job["spec_path"]
    result_file = Path(job["job_dir"]) / "result.txt"
    log_file_path = Path(job["job_dir"]) / "output.log"
    timeout = job["timeout"]
    device = job["device"]
    batch = job["batch"]

    cmd = [
        sys.executable, str(NEURALSAT_MAIN),
        "--net", str(net_path),
        "--spec", str(spec_path),
        "--timeout", str(timeout),
        "--device", device,
        "--batch", str(batch),
        "--verbosity", "0",
        "--result_file", str(result_file),
        "--export_cex",
        "--export_runtime",
    ]

    if job.get("disable_attack"):
        cmd.append("--disable_attack")
    if job.get("disable_restart"):
        cmd.append("--disable_restart")
    if job.get("disable_stabilize"):
        cmd.append("--disable_stabilize")

    force_split = job.get("force_split")
    if force_split in ("input", "hidden"):
        cmd.extend(["--force_split", force_split])

    input_shape = job.get("input_shape")
    if input_shape:
        cmd.append("--input_shape")
        cmd.extend(input_shape)

    output_shape = job.get("output_shape")
    if output_shape:
        cmd.append("--output_shape")
        cmd.extend(output_shape)

    setting_file = job.get("setting_file")
    if setting_file:
        cmd.extend(["--setting_file", str(setting_file)])


    env = os.environ.copy()
    env["PYTHONPATH"] = str(NEURALSAT_ROOT / "src") + ":" + env.get("PYTHONPATH", "")

    try:

        with open(log_file_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(NEURALSAT_ROOT / "src"),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )

            with job_lock:
                job["process"] = proc

            start_time = time.time()
            grace_timeout = timeout + 60
            is_timeout = False

            while proc.poll() is None:
                time.sleep(0.5)

                with job_lock:
                    if job["status"] == "cancelled":
                        break

                if time.time() - start_time > grace_timeout:
                    is_timeout = True
                    break

            if proc.poll() is None:
                if hasattr(os, "killpg") and hasattr(os, "setsid"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()

                if is_timeout:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                else:

                    with job_lock:
                        job["finished"] = time.time()
                    return

        stdout = ""
        if log_file_path.exists():
            stdout = log_file_path.read_text().strip()

        result = "unknown"
        runtime = None
        counterexample = None

        if result_file.exists():
            lines = result_file.read_text().strip().splitlines()
            if lines:
                parts = lines[0].split(",")
                result = parts[0].strip()
                if len(parts) > 1:
                    try:
                        runtime = float(parts[1].strip())
                    except ValueError:
                        pass
                if len(lines) > 1:
                    counterexample = "\n".join(lines[1:])
        else:

            if stdout:
                last_lines = stdout.strip().splitlines()
                if last_lines:
                    last_line = last_lines[-1]
                    parts = last_line.split(",")
                    result = parts[0].strip()
                    if len(parts) > 1:
                        try:
                            runtime = float(parts[1].strip())
                        except ValueError:
                            pass

        with job_lock:
            job["status"] = "completed"
            job["result"] = result
            job["runtime"] = runtime
            job["counterexample"] = counterexample
            job["stdout"] = stdout[-2000:] if stdout else ""
            job["finished"] = time.time()

    except subprocess.TimeoutExpired:
        with job_lock:
            job["status"] = "completed"
            job["result"] = "timeout"
            job["runtime"] = timeout
            job["finished"] = time.time()

    except Exception as e:
        with job_lock:
            job["status"] = "error"
            job["error"] = str(e)
            job["finished"] = time.time()

def _worker(job_id: str):

    with job_lock:
        if jobs[job_id]["status"] == "cancelled":
            return

    with run_lock:

        with job_lock:
            if jobs[job_id]["status"] == "cancelled":
                return
        _run_verification(job_id)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(CLASSIC_DIR, "index.html")

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "neuralsat_root": str(NEURALSAT_ROOT),
        "neuralsat_main_exists": NEURALSAT_MAIN.exists(),
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "running"),
        "queued_jobs": sum(1 for j in jobs.values() if j["status"] == "queued"),
    })

@app.route("/api/verify", methods=["POST"])
def verify():
\
\
\
\
\
\
\
\

    _cleanup_old_jobs()

    if "net" not in request.files:
        return jsonify({"error": "Missing 'net' (ONNX) file"}), 400
    if "spec" not in request.files:
        return jsonify({"error": "Missing 'spec' (VNNLib) file"}), 400

    net_file = request.files["net"]
    spec_file = request.files["spec"]

    if not net_file.filename:
        return jsonify({"error": "Empty ONNX filename"}), 400
    if not spec_file.filename:
        return jsonify({"error": "Empty VNNLib filename"}), 400

    net_name = net_file.filename.lower()
    spec_name = spec_file.filename.lower()
    if not net_name.endswith(".onnx"):
        return jsonify({"error": "Network file must be .onnx format"}), 400
    if not spec_name.endswith(".vnnlib"):
        return jsonify({"error": "Specification file must be .vnnlib format"}), 400

    timeout = min(int(request.form.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)
    device = request.form.get("device", "cuda")
    if device not in ("cpu", "cuda"):
        device = "cuda"
    try:
        batch = int(request.form.get("batch", 1000))
    except (ValueError, TypeError):
        batch = 1000

    disable_attack = request.form.get("disable_attack") == "true"
    disable_restart = request.form.get("disable_restart") == "true"
    disable_stabilize = request.form.get("disable_stabilize") == "true"
    force_split = request.form.get("force_split")

    input_shape, err = _parse_shape(request.form.get("input_shape"))
    if err:
        return jsonify({"error": f"Invalid input_shape: {err}"}), 400

    output_shape, err = _parse_shape(request.form.get("output_shape"))
    if err:
        return jsonify({"error": f"Invalid output_shape: {err}"}), 400

    setting_file, err = _resolve_setting_file(request.form.get("setting_file"))
    if err:
        return jsonify({"error": f"Invalid setting_file: {err}"}), 400
    setting_file = str(setting_file) if setting_file else None

    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    net_path = job_dir / net_file.filename
    spec_path = job_dir / spec_file.filename
    net_file.save(str(net_path))
    spec_file.save(str(spec_path))

    onnx_info = _get_onnx_info(net_path)
    vnnlib_info = _get_vnnlib_info(spec_path)

    job = {
        "id": job_id,
        "status": "queued",
        "created": time.time(),
        "net_name": net_file.filename,
        "spec_name": spec_file.filename,
        "net_path": str(net_path),
        "spec_path": str(spec_path),
        "job_dir": str(job_dir),
        "timeout": timeout,
        "device": device,
        "batch": batch,
        "disable_attack": disable_attack,
        "disable_restart": disable_restart,
        "disable_stabilize": disable_stabilize,
        "force_split": force_split,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "setting_file": setting_file,
        "result": None,
        "runtime": None,
        "counterexample": None,
        "error": None,
        "onnx_info": onnx_info,
        "vnnlib_info": vnnlib_info,
    }

    with job_lock:
        jobs[job_id] = job

    thread = threading.Thread(target=_worker, args=(job_id,), daemon=True)
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "message": f"Verification queued. Use GET /api/status/{job_id} to check progress.",
        "onnx_info": onnx_info,
        "vnnlib_info": vnnlib_info,
    }), 202

@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):

    with job_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job["id"],
        "status": job["status"],
        "net_name": job["net_name"],
        "spec_name": job["spec_name"],
        "timeout": job["timeout"],
        "device": job["device"],
        "onnx_info": job.get("onnx_info"),
        "vnnlib_info": job.get("vnnlib_info"),
    }

    if job["status"] == "running":
        response["elapsed"] = round(time.time() - job.get("started", job["created"]), 1)
    elif job["status"] == "completed":
        response["result"] = job["result"]
        response["runtime"] = job["runtime"]
        response["counterexample"] = job["counterexample"]
    elif job["status"] == "error":
        response["error"] = job["error"]

    return jsonify(response)

@app.route("/api/logs/<job_id>", methods=["GET"])
def get_logs(job_id: str):

    with job_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    log_file_path = Path(job["job_dir"]) / "output.log"
    if not log_file_path.exists():
        return jsonify({"logs": "", "status": job["status"]})

    try:
        logs = log_file_path.read_text()
        return jsonify({
            "logs": logs,
            "status": job["status"]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to read logs: {str(e)}"}), 500

@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel(job_id: str):

    with job_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    with job_lock:
        if job["status"] in ("completed", "error", "cancelled"):
            return jsonify({"status": job["status"], "message": "Job already finished."})

        if job["status"] == "running" and "process" in job:
            proc = job["process"]
            if proc.poll() is None:
                if hasattr(os, "killpg") and hasattr(os, "setsid"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()

        job["status"] = "cancelled"
        job["finished"] = time.time()

    return jsonify({"status": "cancelled", "message": "Job cancelled successfully."})

@app.route("/api/examples", methods=["GET"])
def list_examples():

    onnx_files = []
    vnnlib_files = []

    if EXAMPLE_ONNX_DIR.exists():
        onnx_files = sorted([
            {"name": f.name, "size": f.stat().st_size}
            for f in EXAMPLE_ONNX_DIR.iterdir()
            if f.suffix == ".onnx" and f.stat().st_size < MAX_UPLOAD_BYTES
        ], key=lambda x: x["size"])

    if EXAMPLE_VNNLIB_DIR.exists():
        vnnlib_files = sorted([
            {"name": f.name, "size": f.stat().st_size}
            for f in EXAMPLE_VNNLIB_DIR.iterdir()
            if f.suffix == ".vnnlib"
        ], key=lambda x: x["size"])

    quick_examples = [
        {
            "name": "FNN (tiny, ~1s)",
            "net": "fnn.onnx",
            "spec": "fnn.vnnlib",
            "description": "A small fully-connected network with 2 inputs/outputs.",
        },
        {
            "name": "ACAS Xu 1_1 (prop 6, ~3s)",
            "net": "ACASXU_run2a_1_1_batch_2000.onnx",
            "spec": "prop_6.vnnlib",
            "description": "Aircraft collision avoidance network, property 6 (unsat).",
        },
        {
            "name": "ACAS Xu 1_9 (prop 7, ~6s)",
            "net": "ACASXU_run2a_1_9_batch_2000.onnx",
            "spec": "prop_7.vnnlib",
            "description": "Aircraft collision avoidance network, property 7 (sat).",
        },
    ]

    return jsonify({
        "onnx_files": onnx_files,
        "vnnlib_files": vnnlib_files,
        "quick_examples": quick_examples,
    })

@app.route("/api/example/<path:filename>", methods=["GET"])
def download_example(filename: str):

    filename = Path(filename).name

    onnx_path = EXAMPLE_ONNX_DIR / filename
    if onnx_path.exists() and onnx_path.suffix == ".onnx":
        return send_file(str(onnx_path), as_attachment=True)

    vnnlib_path = EXAMPLE_VNNLIB_DIR / filename
    if vnnlib_path.exists() and vnnlib_path.suffix == ".vnnlib":
        return send_file(str(vnnlib_path), as_attachment=True)

    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    print(f"NeuralSAT root: {NEURALSAT_ROOT}")
    print(f"main.py exists: {NEURALSAT_MAIN.exists()}")
    print(f"Upload dir:     {UPLOAD_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=True)
