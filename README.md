<img width="1920" height="966" alt="image" src="https://github.com/user-attachments/assets/f28039d0-8c46-4763-bc96-a7a0c2db2094" />
<h2 align="center"><i>Home page(Light Theme) </i></h2>
<img width="1920" height="966" alt="image" src="https://github.com/user-attachments/assets/52867db7-9e2b-49aa-94db-a0611bab02fe" />
<h1 align="center"><i>Home page(Dark Theme) </i></h1>
<img width="1920" height="932" alt="image" src="https://github.com/user-attachments/assets/977d582a-78c6-4f58-9147-6017125a0c97" />
<h1 align="center"><i>Results Code Analyzing Page</i></h1>
<img width="1459" height="907" alt="Code_0WTLFR2GfU" src="https://github.com/user-attachments/assets/3b331a06-193e-41b4-b7fc-e7f85ae80150" />
<h1 align="center"><i>Results Dataset Page</i></h1>


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
