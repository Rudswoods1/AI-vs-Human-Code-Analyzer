# AI vs Human Code Analyzer

A minimal Flask web app to compare AI-generated vs Human-written Python code by running:
- Unit tests (pytest)
- Style checks (Pylint)
- Security scan (Bandit)
- Maintainability index (Radon)

UI is clean and minimal (black & white theme) with results shown in cards, a table, and a Chart.js bar chart.

## Requirements
- Python 3.10+

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
flask --app app run
```

Then open the app in your browser at `http://127.0.0.1:5000`.

## Usage
- Upload two Python files (`human.py` and `ai.py`) or paste code into the text areas.
- Click "Сравнить" to run checks. The app creates a temporary workspace, runs pytest (import smoke test), Pylint, Bandit, and Radon MI for each file, and shows results side-by-side.

## Notes
- This prototype uses a simple smoke test (module import) to verify basic correctness. You can expand by adding your own test cases per project.
- The app cleans up temp files after analysis.

---
Created by team Courage and Stupiddy — Askar, Nursultan, and Nurbol
