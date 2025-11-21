AI vs Human Code Analyzer

A Flask-based web application that compares AI-generated Python code with human-written code using multiple metrics:

Pylint — code quality

Bandit — security vulnerabilities

Radon MI — maintainability index

LSTM Neural Network — predicts whether the code was written by an AI model

Final Score — weighted metric combining all checks

UI is clean, minimalistic (Light & Dark themes) and includes:

Results cards

Comparison table

Plotly visualizations

Dataset analysis mode

<img width="1920" height="966" alt="image" src="https://github.com/user-attachments/assets/f28039d0-8c46-4763-bc96-a7a0c2db2094" />
<h2 align="center">Home page(Light Theme)</h2>
<img width="1920" height="966" alt="image" src="https://github.com/user-attachments/assets/52867db7-9e2b-49aa-94db-a0611bab02fe" />
<h2 align="center">Home page(Dark Theme)</h2>
<img width="1920" height="932" alt="image" src="https://github.com/user-attachments/assets/977d582a-78c6-4f58-9147-6017125a0c97" />
<h2 align="center">Results Code Analyzing Page</h2>
<img width="1459" height="907" alt="Code_0WTLFR2GfU" src="https://github.com/user-attachments/assets/3b331a06-193e-41b4-b7fc-e7f85ae80150" />
<h2 align="center">Results Dataset Page</h2>


🚀 Features
✔ Compare AI vs Human code

Upload two files or paste Python code directly.

✔ Automated analysis pipeline:

LSTM prediction (AI / Human)

Code quality check → Pylint

Maintainability index → Radon

Security issues → Bandit

Final score (0–100)

✔ Dataset Mode

Upload a JSON dataset with multiple code pairs — the app:

Analyzes each pair

Generates a CSV file with results

Calculates average metrics

Displays summary charts

✔ Clean UI

Light/Dark themes, modern layout, Plotly charts, and results table.

📦 Requirements

Python 3.10+

Install dependencies:

pip install -r requirements.txt

▶️ Run the App
flask --app app run


Open in browser:

http://127.0.0.1:5000

📝 Usage
1. Code Comparison Mode

Upload two Python files:

human.py

ai.py
OR paste the code into text areas.

Press Сравнить

View results:

Pylint score

Radon MI

Bandit vulnerabilities

LSTM prediction & confidence

Final score

Comparison table

2. Dataset Mode

Go to Dataset Analysis page

Upload JSON file with structure:

[
  {
    "repo": "project-name",
    "path": "file.py",
    "human_code": "...",
    "ai_code": "..."
  }
]


App will:

Analyze each pair

Generate dataset_results.csv

Display aggregated metrics (Human vs AI)

Build summary charts

⚙️ Technologies Used

Flask — backend

Pylint, Bandit, Radon — static analysis

PyTorch — LSTM model

Plotly — charts

HTML / CSS / JS — UI

Chart.js — comparison graphs

🧠 LSTM Model

The project includes an LSTM neural network trained to classify:

AI-generated Python code

Human-written Python code

Key components:

Custom tokenizer

Vocabulary embedding

Binary classification

Output probability + confidence

🧹 Temporary Workspace

Each analysis creates an isolated temp folder:

Saves uploaded code

Runs tools

Cleans up after completion

This prevents conflicts and keeps the app safe.

👥 Authors

Project by team Courage and Stupiddy

Askar

Nurbol

Nursultan

⭐ Want to improve this project?

You can:

Add deeper LLM analysis

Train better tokenizer

Add multi-language support

Improve UI or themes

Extend dataset mode

Pull requests are welcome!
