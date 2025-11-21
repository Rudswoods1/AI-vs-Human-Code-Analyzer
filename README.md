# AI vs Human Code Analyzer

A Flask-based web application that compares **AI-generated Python code**
with **human-written code** using multiple analysis metrics.

------------------------------------------------------------------------

## 🚀 Features

### ✔ AI vs Human Code Comparison

Upload two files or paste Python code directly.

### ✔ Automated Analysis Pipeline

-   **LSTM prediction** --- AI/Human classification
-   **Pylint** --- code quality score
-   **Radon MI** --- maintainability index
-   **Bandit** --- security vulnerabilities
-   **Final Score (0--100)** --- weighted result

### ✔ Dataset Mode

Upload a JSON dataset with multiple code pairs: - Analyzes each pair
- Generates a `dataset_results.csv` file
- Computes average metrics\
- Displays summary statistics

### ✔ Clean Modern UI

-   Light & Dark themes
-   Results cards
-   Comparison table
-   Plotly visualizations

------------------------------------------------------------------------

## 🖼 Screenshots

<h2 align="center"> Home page (Light Theme) </h2>
<img width="1920" src="https://github.com/user-attachments/assets/f28039d0-8c46-4763-bc96-a7a0c2db2094">

<h2 align="center"> Home page (Dark Theme) </h2>
<img width="1920" src="https://github.com/user-attachments/assets/52867db7-9e2b-49aa-94db-a0611bab02fe">

<h2 align="center"> Results: Code Analysis </h2>
<img width="1459" src="https://github.com/user-attachments/assets/977d582a-78c6-4f58-9147-6017125a0c97">`{=html}

<h2 align="center"> Results: Dataset Summary </h2>
<img width="1459" src="https://github.com/user-attachments/assets/3b331a06-193e-41b4-b7fc-e7f85ae80150">

------------------------------------------------------------------------

## 📦 Requirements

-   **Python 3.10+**

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the App

``` bash
flask --app app run
```

Open in your browser:

    http://127.0.0.1:5000

------------------------------------------------------------------------

## 📝 Usage

### **1. Code Comparison Mode**

Upload: - `human.py` - `ai.py`

Or paste code into text fields.

Press **Сравнить**, then view: - Pylint Score\
- Radon MI\
- Bandit Vulnerabilities\
- LSTM Prediction + Confidence\
- Final Score\
- Comparison Table

------------------------------------------------------------------------

### **2. Dataset Mode**

Upload JSON in format:

``` json
[
  {
    "repo": "project-name",
    "path": "file.py",
    "human_code": "...",
    "ai_code": "..."
  }
]
```

App will: - Analyze each pair\
- Generate `dataset_results.csv`\
- Display average metrics\
- Build charts

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   **Flask** --- backend\
-   **Pylint, Bandit, Radon** --- static analysis\
-   **PyTorch** --- LSTM model\
-   **Plotly / Chart.js** --- graphs\
-   **HTML / CSS / JS** --- UI

------------------------------------------------------------------------

## 🧠 LSTM Model

Binary classifier trained to distinguish: - AI-generated code\
- Human-written code

Includes: - Custom tokenizer\
- Vocabulary embeddings\
- LSTM encoder\
- Softmax classifier

------------------------------------------------------------------------

## 🧹 Temporary Workspace

Each task creates an isolated temp folder: - Saves uploaded code\
- Runs analyzers\
- Removes folder afterward

Keeps system clean and secure.

------------------------------------------------------------------------

## 👥 Authors

**Project by team Courage and Stupiddy** - Askar\
- Nurbol\
- Nursultan

------------------------------------------------------------------------

## ⭐ Want to contribute?

Ideas: - Improve tokenizer\
- Add more static analyzers\
- Add multi-language support\
- Enhance UI/themes\
- New LLM evaluation metrics

Pull requests welcome!
