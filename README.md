# Fake Job Posting Detection 🕵️‍♂️

This project provides a machine learning-based prediction system to identify fraudulent job postings. By analyzing textual descriptions and structured categorical data, it uses an XGBoost model to evaluate the risk of a job posting being a scam.

## Features
- **Text Analysis**: Uses TF-IDF to extract meaningful features from job descriptions, requirements, and company profiles.
- **Categorical Handling**: Uses OneHotEncoding for job characteristics like employment type and required experience.
- **XGBoost Classifier**: A highly accurate, tuned model with class-balancing to detect fraudulent postings with a high F1-Score and PR-AUC.
- **Streamlit UI**: An interactive web app that lets you easily paste job details or test pre-loaded samples to get instant predictions and view specific "red flags" for suspicious jobs.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Fake-Job-Posting-Detection
   ```

2. **Create a virtual environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit web application, run the following command in your terminal from the project root:

```bash
streamlit run app.py
```

This will launch the application in your default web browser (typically at `http://localhost:8501`).

## Training the Model (Optional)

If you wish to retrain the model or test other classifiers (Logistic Regression, Random Forest, etc.), you can run the training script. This script will automatically download the dataset if it's missing, train the models, and output evaluation metrics.

```bash
python train.py
```

## Dataset
The project uses the "Real / Fake Job Posting Prediction" dataset (EMSCAD). The training script handles downloading this dataset automatically from a raw open-source mirror.
