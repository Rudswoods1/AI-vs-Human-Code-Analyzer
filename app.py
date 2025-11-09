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


def run_pylint(file_path: Path, cwd: Path) -> Dict[str, Any]:
    """Запускает Pylint для анализа качества кода"""
    cmd = ["pylint", "--score=y", "--disable=R,C", str(file_path.name)]
    code, out, err = run_command(cmd, cwd=cwd, timeout=30)
    score = parse_pylint_score(out + "\n" + err)
    
    # Извлекаем основные ошибки для отображения
    error_msg = None
    full_output = out + "\n" + err
    
    # Ищем строки с ошибками (E) и предупреждениями (W)
    error_lines = []
    for line in full_output.split('\n'):
        if re.search(r':\s*[EWF]\d{4}:', line):
            error_lines.append(line.strip())
    
    if error_lines and len(error_lines) <= 5:
        error_msg = '\n'.join(error_lines[:5])
    elif error_lines:
        error_msg = '\n'.join(error_lines[:3]) + f"\n... and {len(error_lines) - 3} more issues"
    elif code != 0 and score == 0.0:
        error_msg = (err or out).strip()[:500]
    
    return {
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "score": score,
        "error": error_msg,
        "error_count": len(error_lines)
    }


def run_bandit(file_path: Path, cwd: Path) -> Dict[str, Any]:
    """Запускает Bandit для анализа безопасности"""
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
    """Запускает Radon для анализа сложности кода"""
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


def analyze_one(tmp: Path, filename: str) -> dict[str, any]:
    """Анализирует один файл используя все метрики"""
    results: dict[str, any] = {"filename": filename}
    
    # Читаем текст кода
    code_text = (tmp / filename).read_text(encoding="utf-8", errors="replace")

    # LSTM prediction
    try:
        nn_res = predict_lstm(code_text)
        results["ai_label"] = nn_res["label"]
        results["model_confidence"] = nn_res["confidence"]  # в процентах, 0-100
    except Exception as e:
        results["ai_label"] = "Error"
        results["model_confidence"] = 0
        results["lstm_error"] = str(e)

    # Pylint analysis
    pylint_res = run_pylint(Path(filename), cwd=tmp)
    results["pylint_score"] = round(pylint_res.get("score", 0.0), 2)
    results["pylint_error"] = pylint_res.get("error")

    # Radon MI analysis
    radon_res = run_radon_mi(Path(filename), cwd=tmp)
    results["radon_mi"] = round(radon_res.get("mi", 0.0), 2)
    results["radon_error"] = radon_res.get("error")

    # Bandit security analysis
    bandit_res = run_bandit(Path(filename), cwd=tmp)
    results["bandit_vulns"] = bandit_res.get("vulns", 0)
    results["bandit_error"] = bandit_res.get("error")

    # Calculate final quality score with updated weights
    final_score = 0.0
    
    # Pylint 35% (0-10 scaled to 0-35)
    pylint_score = results["pylint_score"]
    final_score += (pylint_score / 10.0) * 35.0
    
    # Radon MI 25% (0-100 scaled to 0-25)
    radon_mi = results["radon_mi"]
    final_score += min(radon_mi, 100.0) * 0.25
    
    # Bandit 30% (0 vulnerabilities = full 30 points)
    if results["bandit_vulns"] == 0:
        final_score += 30.0
    
    # Model Confidence 10% (0-100 scaled to 0-10)
    model_confidence = results.get("model_confidence", 0)
    final_score += (model_confidence / 100.0) * 10.0
    
    results["final_score"] = round(final_score, 2)
    
    print(f"Analysis completed for {filename}: {results}")
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
    """Анализирует два файла: human и ai код"""
    session_dir = Path(tempfile.mkdtemp(prefix=f"analyze_{uuid.uuid4().hex[:8]}_"))
    try:
        h_code = (request.form.get("human_code") or "").strip()
        a_code = (request.form.get("ai_code") or "").strip()

        # Обработка Human файла
        if "human_file" in request.files and request.files["human_file"].filename:
            ok, _, msg = save_upload_to(session_dir, "human_file", "human.py")
            if not ok:
                flash(f"Human file error: {msg}", "danger")
                return redirect(url_for("index"))
        else:
            if not h_code:
                flash("Please provide Human code or upload a file.", "warning")
                return redirect(url_for("index"))
            write_code_to_file(session_dir, "human.py", h_code)

        # Обработка AI файла
        if "ai_file" in request.files and request.files["ai_file"].filename:
            ok, _, msg = save_upload_to(session_dir, "ai_file", "ai.py")
            if not ok:
                flash(f"AI file error: {msg}", "danger")
                return redirect(url_for("index"))
        else:
            if not a_code:
                flash("Please provide AI code or upload a file.", "warning")
                return redirect(url_for("index"))
            write_code_to_file(session_dir, "ai.py", a_code)

        # Анализируем оба файла
        human_result = analyze_one(session_dir, "human.py")
        ai_result = analyze_one(session_dir, "ai.py")

        # Создаем список результатов для шаблона
        results = [human_result, ai_result]
        
        return render_template("results.html", results=results)
    finally:
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(debug=True)
