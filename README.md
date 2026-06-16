# TER Tabular XAI App

This application is an interactive **Streamlit** tool designed for tabular data exploration, advanced machine learning model training (using the TALENT library), and model explainability (XAI).

It integrates several modules:
- **Exploratory Data Analysis (EDA)**: Data distribution visualization, correlations, and bivariate analysis.
- **Data Management & Transformation**: Data cleaning, encoding, and preparation for training.
- **Model Training**: Support for state-of-the-art tabular models (TabNet, TabR, TabPFN, FTT, ResNet, etc.).
- **Explainability (XAI)**: Global explanation methods (Permutation Importance) and local methods (SHAP, LIME) with an integrated LLM assistant for interpretation.

---

## 🚀 How to Run the Application

Follow these steps to launch the application in any environment (Windows, macOS, or Linux).

### 1. Prerequisites
Make sure you have **Python 3.8+** (ideally 3.9 or 3.10) installed on your machine.

### 2. Clone or Download the Project
Navigate to the directory containing the application files (the folder containing `app.py`).

### 3. Create a Virtual Environment (Recommended)
It is highly recommended to create a virtual environment to isolate the project dependencies.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*(Optional)* If you encounter PyTorch-related issues during installation, make sure to install the CPU version of PyTorch compatible with your system via the official PyTorch website.

### 5. Launch the Application
Run the following command to start the Streamlit application:
```bash
streamlit run app.py
```

Your default web browser should automatically open to the local address (usually `http://localhost:8501`). If it doesn't, you can click on the link displayed in your terminal.
