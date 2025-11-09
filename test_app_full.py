import io
import os
import json
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import app  # импорт твоего основного файла


# -----------------------------
# 1. ТЕСТЫ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ
# -----------------------------

def test_allowed_file():
    """Проверка валидации расширений файлов"""
    assert app.allowed_file("main.py")
    assert app.allowed_file("script.py")
    assert not app.allowed_file("main.txt")
    assert not app.allowed_file("script")
    assert not app.allowed_file("file.js")


def test_normalize_code_text():
    """Проверка нормализации переводов строк"""
    code = "print('hi')\r\nprint('bye')\rprint('end')"
    normalized = app.normalize_code_text(code)
    assert "\r" not in normalized
    assert normalized.count("\n") == 2
    assert "print('hi')" in normalized


def test_write_code_to_file(tmp_path):
    """Проверка записи кода в файл"""
    code = "def test():\n    print('ok')"
    file_path = app.write_code_to_file(tmp_path, "test.py", code)
    
    assert file_path.exists()
    assert file_path.name == "test.py"
    content = file_path.read_text()
    assert "def test()" in content
    assert "print('ok')" in content


# -----------------------------
# 2. ТЕСТЫ ТОКЕНИЗАЦИИ И LSTM
# -----------------------------

def test_simple_tokenizer():
    """Проверка токенизации кода"""
    code = "def add(a, b): return a+b"
    tokens = app.simple_tokenizer(code)
    
    assert isinstance(tokens, list)
    assert "def" in tokens
    assert "add" in tokens
    assert len(tokens) <= app.MAX_LEN


def test_simple_tokenizer_long_code():
    """Проверка токенизации длинного кода (обрезка)"""
    long_code = "x = 1\n" * 5000
    tokens = app.simple_tokenizer(long_code)
    
    assert len(tokens) == app.MAX_LEN


def test_encode():
    """Проверка кодирования токенов в индексы"""
    tokens = ["def", "add", "(", "a", ",", "b", ")"]
    ids = app.encode(tokens)
    
    assert isinstance(ids, list)
    assert len(ids) == app.MAX_LEN
    assert all(isinstance(i, int) for i in ids)


def test_encode_padding():
    """Проверка паддинга коротких последовательностей"""
    tokens = ["def", "test"]
    ids = app.encode(tokens)
    
    assert len(ids) == app.MAX_LEN
    # Проверяем, что есть нули (паддинг)
    assert 0 in ids


def test_predict_lstm():
    """Проверка предсказания LSTM модели"""
    code = "def add(a, b):\n    return a + b"
    result = app.predict_lstm(code)
    
    assert "label" in result
    assert "confidence" in result
    # Модель может возвращать разные метки
    assert result["label"] in ["AI", "Human", "AI_generated", "Human_written"]
    assert 0 <= result["confidence"] <= 100


# -----------------------------
# 3. ТЕСТЫ ВЫПОЛНЕНИЯ КОМАНД
# -----------------------------

def test_run_command_success():
    """Проверка успешного выполнения команды"""
    code, out, err = app.run_command(
        ["python", "-c", "print('Hello')"], 
        cwd=Path("."), 
        timeout=5
    )
    
    assert code == 0
    assert "Hello" in out


def test_run_command_timeout():
    """Проверка таймаута команды"""
    code, out, err = app.run_command(
        ["python", "-c", "import time; time.sleep(10)"], 
        Path("."), 
        timeout=1
    )
    
    assert code == 124  # код таймаута


def test_run_command_error():
    """Проверка обработки ошибки команды"""
    code, out, err = app.run_command(
        ["python", "-c", "raise Exception('test error')"], 
        Path("."), 
        timeout=5
    )
    
    assert code != 0


# -----------------------------
# 4. ТЕСТЫ ПАРСИНГА РЕЗУЛЬТАТОВ
# -----------------------------

def test_parse_pylint_score():
    """Проверка парсинга оценки pylint"""
    output = "Your code has been rated at 8.23/10 (previous run: 7.50/10, +0.73)"
    score = app.parse_pylint_score(output)
    assert abs(score - 8.23) < 0.01
    
    assert app.parse_pylint_score("no score here") == 0.0
    assert app.parse_pylint_score("") == 0.0


def test_parse_pylint_score_edge_cases():
    """Проверка edge cases для парсинга pylint"""
    assert app.parse_pylint_score("rated at 10.00/10") == 10.0
    assert app.parse_pylint_score("rated at 0.00/10") == 0.0
    assert abs(app.parse_pylint_score("rated at 5.5/10") - 5.5) < 0.01


# -----------------------------
# 5. ТЕСТЫ ЭВРИСТИЧЕСКИХ МЕТРИК
# -----------------------------

def test_compute_heuristics():
    """Проверка вычисления эвристических метрик"""
    code = """def add_numbers(a, b):
    # Функция сложения
    result = a + b
    return result
"""
    metrics = app.compute_heuristics(code)
    
    assert "function_naming" in metrics
    assert "variable_naming" in metrics
    assert "indentation" in metrics
    assert "comment_ratio" in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 100 for v in metrics.values())


def test_compute_heuristics_snake_case():
    """Проверка распознавания snake_case"""
    code = """def my_function():
    my_variable = 10
    another_var = 20
"""
    metrics = app.compute_heuristics(code)
    
    assert metrics["function_naming"] > 0
    assert metrics["variable_naming"] > 0


def test_compute_heuristics_camel_case():
    """Проверка обработки camelCase (не snake_case)"""
    code = """def myFunction():
    myVariable = 10
"""
    metrics = app.compute_heuristics(code)
    
    # camelCase не должен считаться snake_case
    assert metrics["function_naming"] < 100


def test_detect_ai_like():
    """Проверка детекции AI-подобного кода"""
    metrics = {
        "function_naming": 90.0,
        "variable_naming": 85.0,
        "indentation": 95.0,
        "comment_ratio": 20.0
    }
    result = app.detect_ai_like(metrics)
    
    assert "ai_like" in result
    assert "human_like" in result
    assert "confidence" in result
    assert 0 <= result["ai_like"] <= 100
    assert 0 <= result["human_like"] <= 100
    assert abs(result["ai_like"] + result["human_like"] - 100.0) < 0.1


# -----------------------------
# 6. ТЕСТЫ АНАЛИЗА ФАЙЛОВ
# -----------------------------

def test_run_pylint(tmp_path):
    """Проверка запуска pylint"""
    code = "def add(a, b):\n    return a + b\n"
    file_path = tmp_path / "test.py"
    file_path.write_text(code)
    
    result = app.run_pylint(file_path, cwd=tmp_path)
    
    assert "returncode" in result
    assert "score" in result
    assert isinstance(result["score"], float)


def test_run_bandit(tmp_path):
    """Проверка запуска bandit"""
    code = "def safe_function():\n    return 42\n"
    file_path = tmp_path / "test.py"
    file_path.write_text(code)
    
    result = app.run_bandit(file_path, cwd=tmp_path)
    
    assert "returncode" in result
    assert "vulns" in result
    assert isinstance(result["vulns"], int)


def test_run_radon_mi(tmp_path):
    """Проверка запуска radon maintainability index"""
    code = "def simple():\n    return 1\n"
    file_path = tmp_path / "test.py"
    file_path.write_text(code)
    
    result = app.run_radon_mi(file_path, cwd=tmp_path)
    
    assert "returncode" in result
    assert "mi" in result
    assert isinstance(result["mi"], float)


# -----------------------------
# 7. ТЕСТЫ ОСНОВНОЙ ФУНКЦИИ АНАЛИЗА
# -----------------------------

def test_analyze_one(tmp_path, monkeypatch):
    """Проверка полного анализа файла"""
    code = """def add(a, b):
    # Сложение двух чисел
    return a + b
"""
    file_path = tmp_path / "test.py"
    file_path.write_text(code)
    
    result = app.analyze_one(tmp_path, "test.py")
    
    assert "filename" in result
    assert result["filename"] == "test.py"
    assert "heuristics" in result
    assert "ai_detection" in result
    assert "pylint_score" in result
    assert "tests_passed" in result
    assert "tests_failed" in result
    assert "bandit_vulns" in result
    assert "radon_mi" in result


def test_analyze_one_with_lstm(tmp_path):
    """Проверка анализа с LSTM предсказанием"""
    code = "def multiply(x, y):\n    return x * y\n"
    file_path = tmp_path / "calc.py"
    file_path.write_text(code)
    
    result = app.analyze_one(tmp_path, "calc.py")
    
    assert "lstm_label" in result or "lstm_error" in result
    if "lstm_label" in result:
        # Модель может возвращать разные метки
        assert result["lstm_label"] in ["AI", "Human", "AI_generated", "Human_written"]
        assert "lstm_confidence" in result


# -----------------------------
# 8. ТЕСТЫ FLASK API
# -----------------------------

@pytest.fixture
def client():
    """Создание тестового клиента Flask"""
    app.app.config["TESTING"] = True
    app.app.config["WTF_CSRF_ENABLED"] = False
    with app.app.test_client() as client:
        yield client


def test_index_route(client):
    """Проверка главной страницы"""
    res = client.get("/")
    assert res.status_code == 200
    assert b"html" in res.data or b"HTML" in res.data


def test_about_route(client):
    """Проверка страницы About"""
    res = client.get("/about")
    assert res.status_code == 200


def test_dark_route(client):
    """Проверка темной темы страницы"""
    res = client.get("/dark")
    assert res.status_code == 200


def test_analyze_route_text_input(client, tmp_path, monkeypatch):
    """Проверка анализа через текстовый ввод"""
    # Мокаем analyze_one для быстрого тестирования
    mock_result = {
        "filename": "test.py",
        "heuristics": {"function_naming": 80.0},
        "ai_detection": {"ai_like": 60.0},
        "pylint_score": 8.0,
        "tests_passed": 1,
        "tests_failed": 0,
        "bandit_vulns": 0,
        "radon_mi": 75.0,
        "lstm_label": "AI",
        "lstm_confidence": 85.5
    }
    
    monkeypatch.setattr(app, "analyze_one", lambda d, f: mock_result)
    
    data = {
        "human_code": "def h():\n    pass",
        "ai_code": "def a():\n    pass"
    }
    
    res = client.post("/analyze", data=data, follow_redirects=True)
    assert res.status_code == 200


def test_analyze_route_no_code(client):
    """Проверка обработки пустого запроса"""
    res = client.post("/analyze", data={}, follow_redirects=True)
    assert res.status_code == 200


def test_analyze_route_file_upload(client, tmp_path, monkeypatch):
    """Проверка загрузки файлов"""
    mock_result = {
        "filename": "human.py",
        "heuristics": {},
        "ai_detection": {},
        "pylint_score": 0.0,
        "tests_passed": 0,
        "tests_failed": 0
    }
    
    monkeypatch.setattr(app, "analyze_one", lambda d, f: mock_result)
    
    data = {
        "human_file": (io.BytesIO(b"def test(): pass"), "human.py"),
        "ai_file": (io.BytesIO(b"def ai(): pass"), "ai.py"),
    }
    
    res = client.post("/analyze", data=data, follow_redirects=True)
    assert res.status_code == 200


# -----------------------------
# 9. ТЕСТЫ МОДЕЛИ
# -----------------------------

def test_model_structure():
    """Проверка структуры LSTM модели"""
    model = app.CodeClassifier(vocab_size=100)
    x = torch.randint(0, 100, (2, app.MAX_LEN))  # batch_size=2
    
    out = model(x)
    
    assert out.shape == (2, 2)  # (batch_size, num_classes)
    assert not torch.isnan(out).any()


def test_model_forward_pass():
    """Проверка forward pass модели"""
    model = app.CodeClassifier(vocab_size=50, embed_dim=64, hidden_dim=128)
    model.eval()
    
    x = torch.randint(0, 50, (1, app.MAX_LEN))
    
    with torch.no_grad():
        out = model(x)
    
    assert out.shape == (1, 2)
    assert torch.all(torch.isfinite(out))


# -----------------------------
# 10. ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# -----------------------------

def test_full_pipeline(tmp_path):
    """Проверка полного пайплайна анализа"""
    human_code = """def calculate_sum(numbers):
    # Calculate sum of numbers
    total = 0
    for num in numbers:
        total += num
    return total
"""
    
    ai_code = """def calculate_product(numbers):
    # Calculate product of numbers
    result = 1
    for num in numbers:
        result *= num
    return result
"""
    
    human_file = tmp_path / "human.py"
    ai_file = tmp_path / "ai.py"
    human_file.write_text(human_code)
    ai_file.write_text(ai_code)
    
    h_result = app.analyze_one(tmp_path, "human.py")
    a_result = app.analyze_one(tmp_path, "ai.py")
    
    assert h_result["filename"] == "human.py"
    assert a_result["filename"] == "ai.py"
    assert "heuristics" in h_result
    assert "ai_detection" in a_result


# -----------------------------
# 11. ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ
# -----------------------------

try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
def test_performance_compute_heuristics(benchmark):
    """Бенчмарк вычисления эвристик на РЕАЛЬНОМ коде"""
    # Используем реальные функции из вашего app.py
    realistic_code = '''
def allowed_file(filename: str) -> bool:
    """Проверка расширения файла"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_code_text(code: str) -> str:
    """Нормализация переводов строк"""
    code = code.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    return code

def write_code_to_file(base_dir: Path, filename: str, code: str) -> Path:
    """Запись кода в файл"""
    path = base_dir / filename
    path.write_text(normalize_code_text(code), encoding="utf-8")
    return path

def run_command(cmd: list, cwd: Path, timeout: int = 20, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Выполнение команды с таймаутом"""
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
'''
    
    # Дублируем для создания большого объема кода (имитируем анализ большого проекта)
    large_realistic_code = realistic_code * 25  # 25 копий реального кода
    
    result = benchmark(app.compute_heuristics, large_realistic_code)
    assert "function_naming" in result
    assert "variable_naming" in result
    assert "indentation" in result
    assert "comment_ratio" in result


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
def test_performance_tokenizer(benchmark):
    """Бенчмарк токенизации на РЕАЛЬНОМ коде LSTM модели"""
    # Используем реальный код из вашего LSTM-классификатора
    realistic_code = '''
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

def simple_tokenizer(code):
    code = code.lower()
    tokens = re.findall(r"[a-zA-Z_]+|\\S", code)
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
'''
    
    # Дублируем для создания большого файла (имитируем анализ большого модуля)
    large_realistic_code = realistic_code * 20
    
    result = benchmark(app.simple_tokenizer, large_realistic_code)
    assert isinstance(result, list)
    assert len(result) > 0
    # Проверяем, что токенизация работает корректно
    assert all(isinstance(token, str) for token in result)

'''
IQR

round_times = [0.0015, 0.0014, 0.0016, 0.0013, 0.0018, 
               0.0014, 0.0015, 0.0017, 0.0012, 0.0019]

sorted_times = sorted(round_times)
n = 10
q1_pos = 10 * 0.25 = 2.5  # между 2-м и 3-м элементами
q3_pos = 10 * 0.75 = 7.5  # между 7-м и 8-м элементами

q1 = 0.0014 + 0.5 * (0.0014 - 0.0014) = 0.0014
q3 = 0.0017 + 0.5 * (0.0018 - 0.0017) = 0.00175

iqr = 0.00175 - 0.0014 = 0.00035

'''
# -----------------------------
# 12. EDGE CASES
# -----------------------------

def test_empty_code():
    """Проверка обработки пустого кода"""
    metrics = app.compute_heuristics("")
    assert all(v == 0.0 for v in metrics.values())


def test_unicode_code(tmp_path):
    """Проверка обработки unicode символов"""
    code = "def тест():\n    # Комментарий\n    return 'привет'"
    file_path = tmp_path / "unicode.py"
    file_path.write_text(code, encoding="utf-8")
    
    result = app.analyze_one(tmp_path, "unicode.py")
    assert result["filename"] == "unicode.py"


def test_very_long_code():
    """Проверка обработки очень длинного кода"""
    code = "x = 1\n" * 10000
    tokens = app.simple_tokenizer(code)
    assert len(tokens) == app.MAX_LEN
