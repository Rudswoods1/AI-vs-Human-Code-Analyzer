import io
import os
import json
import time
import tempfile
import types
import pytest
import torch
from pathlib import Path

import app  # импорт твоего основного файла


# -----------------------------
# 1. ТЕСТЫ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ
# -----------------------------

def test_allowed_file():
    assert app.allowed_file("main.py")
    assert not app.allowed_file("main.txt")
    assert not app.allowed_file("script")

def test_normalize_code_text():
    code = "print('hi')\r\nprint('bye')"
    normalized = app.normalize_code_text(code)
    assert "\r" not in normalized
    assert "\n" in normalized

def test_write_code_to_file(tmp_path):
    file_path = app.write_code_to_file(tmp_path, "test.py", "print('ok')")
    assert file_path.exists()
    assert "print" in file_path.read_text()

# -----------------------------
# 2. ТЕСТЫ ЛОГИКИ АНАЛИЗА КОДА
# -----------------------------

def test_simple_tokenizer():
    tokens = app.simple_tokenizer("def add(a, b): return a+b")
    assert isinstance(tokens, list)
    assert "def" in tokens

def test_encode_length():
    tokens = ["print", "(", "1", ")"]
    ids = app.encode(tokens)
    assert isinstance(ids, list)
    assert len(ids) == app.MAX_LEN

def test_compute_heuristics_and_detect_ai_like():
    code = "def add_numbers(a, b):\n    return a + b\n"
    metrics = app.compute_heuristics(code)
    result = app.detect_ai_like(metrics)
    assert "ai_like" in result
    assert 0 <= result["ai_like"] <= 100

# -----------------------------
# 3. ТЕСТЫ ОСНОВНЫХ КОМАНДНЫХ ЗАПУСКОВ
# -----------------------------

def test_run_command_success():
    code, out, err = app.run_command(["echo", "Hello"], cwd=Path("."), timeout=5)
    assert code == 0
    assert "Hello" in out

def test_run_command_timeout():
    code, out, err = app.run_command(["python", "-c", "import time; time.sleep(2)"], Path("."), timeout=1)
    assert code == 124  # таймаут возвращает 124

# -----------------------------
# 4. ТЕСТЫ ДЛЯ ФУНКЦИЙ С АНАЛИЗОМ ФАЙЛОВ
# -----------------------------

def test_parse_pylint_score():
    score = app.parse_pylint_score("Your code has been rated at 8.23/10")
    assert abs(score - 8.23) < 0.01
    assert app.parse_pylint_score("no score") == 0.0

# -----------------------------
# 5. ПРОВЕРКА analyze_one (основная функция анализа)
# -----------------------------

def test_analyze_one(tmp_path, monkeypatch):
    # создаём фейковый файл
    code = "def add(a,b):\n    return a+b"
    (tmp_path / "human.py").write_text(code)

    # подменяем predict_lstm (чтобы не грузить модель)
    monkeypatch.setattr(app, "predict_lstm", lambda c: {"label": "AI", "confidence": 95.0})

    result = app.analyze_one(tmp_path, "human.py")
    assert "heuristics" in result
    assert "ai_detection" in result
    assert "pylint_score" in result

# -----------------------------
# 6. ПРОВЕРКА FLASK API
# -----------------------------

@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client

def test_index_route(client):
    res = client.get("/")
    assert res.status_code == 200

def test_about_route(client):
    res = client.get("/about")
    assert res.status_code == 200

def test_analyze_route_text_input(client, monkeypatch):
    # подменим анализ, чтобы не вызывались внешние утилиты
    monkeypatch.setattr(app, "analyze_one", lambda d, f: {"fake": True})
    data = {
        "human_code": "def h(): pass",
        "ai_code": "def a(): pass"
    }
    res = client.post("/analyze", data=data, follow_redirects=True)
    assert res.status_code == 200
    assert b"results" in res.data or b"fake" in res.data

# -----------------------------
# 7. ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ
# -----------------------------

def test_performance_compute_heuristics(benchmark):
    code = "def f():\n    pass\n" * 1000
    benchmark(app.compute_heuristics, code)

# -----------------------------
# 8. ПРОВЕРКА ЗАГРУЗКИ LSTM-МОДЕЛИ
# -----------------------------

def test_model_structure():
    model = app.CodeClassifier(vocab_size=10)
    x = torch.randint(0, 10, (1, 50))
    out = model(x)
    assert out.shape[-1] == 2