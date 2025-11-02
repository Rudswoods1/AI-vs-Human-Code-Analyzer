import os
import re
import json
import shutil
import tempfile
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- LSTM Model Integration ---
import torch
import torch.nn as nn
import numpy as np
import re

MODEL_PATH = r"C:\Users\nurbolik\Downloads\Telegram Desktop\AI-vs-Human-Code-Analyzer-main-with-ai\AI-vs-Human-Code-Analyzer-main\ai_code_detector_lstm.pth"
MAX_LEN = 3000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Модель как в обучении
class CodeClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = self.dropout(h[-1])
        return self.fc(out)


# Загрузка модели
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
vocab = checkpoint["vocab"]
label_classes = checkpoint["label_encoder"]
model = CodeClassifier(len(vocab)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()


def simple_tokenizer(code):
    code = code.lower()
    tokens = re.findall(r"[a-zA-Z_]+|\S", code)
    return tokens[:MAX_LEN]


def encode(tokens):
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return ids[:MAX_LEN]


def predict_lstm(code_text: str):
    tokens = simple_tokenizer(code_text)
    ids = encode(tokens)
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_classes[pred_idx]
    confidence = float(np.max(probs) * 100)
    return {"label": pred_label, "confidence": round(confidence, 2)}


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

ALLOWED_EXTENSIONS = {"py"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB per file
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_code_text(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code


def write_code_to_file(base_dir: Path, filename: str, code: str) -> Path:
    path = base_dir / filename
    path.write_text(normalize_code_text(code), encoding="utf-8")
    return path


def save_upload_to(
    base_dir: Path, field_name: str, dest_filename: str
) -> Tuple[bool, Path, str]:
    file = request.files.get(field_name)
    if not file or file.filename == "":
        return False, Path(), "No file provided"
    if not allowed_file(file.filename):
        return False, Path(), "Only .py files are allowed"
    filename = secure_filename(dest_filename)
    dest_path = base_dir / filename
    file.save(dest_path)
    try:
        txt = dest_path.read_text(encoding="utf-8", errors="replace")
        dest_path.write_text(normalize_code_text(txt), encoding="utf-8")
    except Exception:
        pass
    return True, dest_path, ""


def run_command(
    cmd: list, cwd: Path, timeout: int = 20, env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    try:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=merged_env,
            shell=(os.name == "nt")
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or "Timed out")
    except Exception as e:
        return 1, "", str(e)


def parse_pylint_score(output: str) -> float:
    match = re.search(r"rated at\s+([0-9]+(?:\.[0-9]+)?)/10", output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def run_pytest_for_module(tmp: Path, module_filename: str) -> Dict[str, Any]:
    test_code = f"""
import importlib
import pathlib
import sys
import builtins
import io
import os as _os

# Make project importable
sys.path.insert(0, str(pathlib.Path('.').resolve()))

# Prepare safe stubs
builtins.input = lambda *a, **k: ""
try:
    import getpass
    getpass.getpass = lambda *a, **k: ""
except Exception:
    pass
sys.stdin = io.StringIO("")
_sys_exit = sys.exit
sys.exit = lambda *a, **k: None
_os_system = _os.system
_os.system = lambda *a, **k: 0


def test_import():
    try:
        modname = '{Path(module_filename).stem}'
        mod = importlib.import_module(modname)
        assert mod is not None
    finally:
        sys.exit = _sys_exit
        _os.system = _os_system


def test_add_if_exists():
    modname = '{Path(module_filename).stem}'
    mod = importlib.import_module(modname)
    if hasattr(mod, 'add') and callable(getattr(mod, 'add')):
        assert mod.add(2, 3) == 5


def test_main_if_exists():
    modname = '{Path(module_filename).stem}'
    mod = importlib.import_module(modname)
    if hasattr(mod, 'main') and callable(getattr(mod, 'main')):
        res = mod.main()
        assert res is None or isinstance(res, int)
"""
    test_path = tmp / "test_smoke.py"
    test_path.write_text(test_code, encoding="utf-8")
    code, out, err = run_command(
        ["pytest", "-q", str(test_path.name)],
        cwd=tmp,
        timeout=25,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    passed = 0
    failed = 0
    m = re.search(r"(\d+)\s+passed", out)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", out)
    if m:
        failed = int(m.group(1))
    error_msg = None
    if code != 0 and passed == 0 and failed == 0:
        snippet = (out + "\n" + err).strip()
        error_msg = "\n".join(snippet.splitlines()[:30])[:1500]
    return {
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "passed": passed,
        "failed": failed,
        "error": error_msg,
    }


def run_pylint(file_path: Path, cwd: Path) -> Dict[str, Any]:
    cmd = ["pylint", "--score=y", "--disable=R,C", str(file_path.name)]
    code, out, err = run_command(cmd, cwd=cwd, timeout=30)
    score = parse_pylint_score(out + "\n" + err)
    error_msg = None
    if code != 0 and score == 0.0:
        error_msg = (err or out).strip()[:500]
    return {
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "score": score,
        "error": error_msg,
    }


def run_bandit(file_path: Path, cwd: Path) -> Dict[str, Any]:
    cmd = ["bandit", "-q", "-f", "json", "-lll", "-iii", str(file_path.name)]
    code, out, err = run_command(cmd, cwd=cwd, timeout=30)
    vulns = 0
    try:
        data = json.loads(out or "{}")
        results = data.get("results", [])
        if results:
            vulns = len(
                [r for r in results if r.get("filename", "").endswith(file_path.name)]
            )
    except json.JSONDecodeError:
        pass
    error_msg = None
    if code != 0 and vulns == 0:
        error_msg = (err or out).strip()[:500]
    return {
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "vulns": vulns,
        "error": error_msg,
    }


def run_radon_mi(file_path: Path, cwd: Path) -> Dict[str, Any]:
    cmd = ["radon", "mi", "-j", str(file_path.name)]
    code, out, err = run_command(cmd, cwd=cwd, timeout=25)
    mi = 0.0
    try:
        data = json.loads(out or "{}")
        file_key = str(file_path.name)
        if file_key in data and isinstance(data[file_key], dict):
            mi = float(data[file_key].get("mi", 0.0))
        else:
            vals = list(data.values())
            if vals and isinstance(vals[0], dict):
                mi = float(vals[0].get("mi", 0.0))
    except Exception:
        pass
    error_msg = None
    if code != 0 and mi == 0.0:
        error_msg = (err or out).strip()[:500]
    return {
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "mi": mi,
        "error": error_msg,
    }


# --- Heuristic metrics ---
IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
CAMEL_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$")


def compute_heuristics(code_text: str) -> Dict[str, float]:
    lines = [ln for ln in code_text.split("\n")]
    non_empty = [ln for ln in lines if ln.strip()]
    # Comment ratio
    comments = [ln for ln in lines if ln.strip().startswith("#")]
    comment_ratio = (len(comments) / max(1, len(non_empty))) * 100.0
    # Indentation: rough share of lines starting with 4-space multiples
    indent_ok = 0
    for ln in non_empty:
        m = re.match(r"^(\s+)", ln)
        if m:
            sp = m.group(1)
            spaces = sp.count(" ")
            tabs = sp.count("\t")
            if tabs == 0 and (spaces % 4 == 0):
                indent_ok += 1
    indent_pct = (indent_ok / max(1, len(non_empty))) * 100.0
    # Function and variable naming
    func_names = re.findall(
        r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\(", code_text, flags=re.M
    )
    var_names = re.findall(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", code_text, flags=re.M)
    func_snake = sum(
        1 for n in func_names if IDENTIFIER_RE.match(n) and not CAMEL_RE.match(n)
    )
    var_snake = sum(
        1 for n in var_names if IDENTIFIER_RE.match(n) and not CAMEL_RE.match(n)
    )
    func_pct = (func_snake / max(1, len(func_names))) * 100.0 if func_names else 0.0
    var_pct = (var_snake / max(1, len(var_names))) * 100.0 if var_names else 0.0
    return {
        "function_naming": round(func_pct, 1),
        "variable_naming": round(var_pct, 1),
        "indentation": round(indent_pct, 1),
        "comment_ratio": round(comment_ratio, 1),
    }


def detect_ai_like(metrics: Dict[str, float]) -> Dict[str, float]:
    # Extremely simple heuristic: more consistent naming + moderate indentation + moderate comments => more AI-like
    score = 0.0
    score += metrics.get("function_naming", 0) * 0.3
    score += metrics.get("variable_naming", 0) * 0.3
    score += metrics.get("indentation", 0) * 0.2
    cr = metrics.get("comment_ratio", 0)
    score += (100.0 - abs(cr - 20.0)) * 0.2 / 100.0 * 100.0  # peak around 20%
    # clamp
    ai_like = max(0.0, min(100.0, score))
    human_like = round(100.0 - ai_like, 1)
    confidence = round(min(100.0, 50.0 + abs(ai_like - 50.0)), 1)
    return {
        "ai_like": round(ai_like, 1),
        "human_like": human_like,
        "confidence": confidence,
    }


def analyze_one(tmp: Path, filename: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {"filename": filename}
    # Read code text for heuristics
    code_text = (tmp / filename).read_text(encoding="utf-8", errors="replace")

    # LSTM prediction
    try:
        nn_res = predict_lstm(code_text)
        results["lstm_label"] = nn_res["label"]
        results["lstm_confidence"] = nn_res["confidence"]
    except Exception as e:
        results["lstm_error"] = str(e)

    m = compute_heuristics(code_text)
    d = detect_ai_like(m)
    results["heuristics"] = m
    results["ai_detection"] = d

    pytest_res = run_pytest_for_module(tmp, filename)
    results["tests_passed"] = pytest_res.get("passed", 0)
    results["tests_failed"] = pytest_res.get("failed", 0)
    results["pytest_error"] = pytest_res.get("error")

    pylint_res = run_pylint(Path(filename), cwd=tmp)
    results["pylint_score"] = round(pylint_res.get("score", 0.0), 2)
    results["pylint_error"] = pylint_res.get("error")

    bandit_res = run_bandit(Path(filename), cwd=tmp)
    results["bandit_vulns"] = bandit_res.get("vulns", 0)
    results["bandit_error"] = bandit_res.get("error")

    radon_res = run_radon_mi(Path(filename), cwd=tmp)
    results["radon_mi"] = round(radon_res.get("mi", 0.0), 2)
    results["radon_error"] = radon_res.get("error")
    print(results)

    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/dark')
def dark():
    return render_template('dark.html')


@app.route("/analyze", methods=["POST"])
def analyze():
    session_dir = Path(tempfile.mkdtemp(prefix=f"analyze_{uuid.uuid4().hex[:8]}_"))
    try:
        h_code = (request.form.get("human_code") or "").strip()
        a_code = (request.form.get("ai_code") or "").strip()

        if "human_file" in request.files and request.files["human_file"].filename:
            ok, _, msg = save_upload_to(session_dir, "human_file", "human.py")
            if not ok:
                flash(f"Human file error: {msg}", "danger")
                return redirect(url_for("index"))
        else:
            if not h_code:
                flash("Provide Human code or upload a file.", "warning")
                return redirect(url_for("index"))
            write_code_to_file(session_dir, "human.py", h_code)

        if "ai_file" in request.files and request.files["ai_file"].filename:
            ok, _, msg = save_upload_to(session_dir, "ai_file", "ai.py")
            if not ok:
                flash(f"AI file error: {msg}", "danger")
                return redirect(url_for("index"))
        else:
            if not a_code:
                flash("Provide AI code or upload a file.", "warning")
                return redirect(url_for("index"))
            write_code_to_file(session_dir, "ai.py", a_code)

        h_res = analyze_one(session_dir, "human.py")
        a_res = analyze_one(session_dir, "ai.py")

        results = {"human": h_res, "ai": a_res}
        return render_template("results.html", results=results)
    finally:
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(debug=True)
