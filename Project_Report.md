# Project Report: Fake Job Posting Detection using Machine Learning

## 1. Abstract
The proliferation of fraudulent job postings online poses a significant threat to job seekers, leading to financial loss, identity theft, and a loss of trust in digital employment platforms. This project aims to develop a robust machine learning system capable of identifying fake job advertisements. By combining unstructured textual data (e.g., job descriptions, company profiles) with structured categorical features (e.g., employment type, telecommuting status), we trained and evaluated multiple classification models. The final system, powered by an XGBoost classifier, achieved an F1-Score of 0.8529 and was deployed as an interactive Streamlit web application featuring a Cyber Threat Dashboard, interactive Plotly Radar charts, and enterprise-grade Batch CSV Analysis capabilities.

## 2. Introduction
### 2.1 Problem Statement
Online job portals often struggle to filter out malicious actors who post fake job listings to harvest personal data or execute financial scams. Manually reviewing thousands of postings is unscalable. 

### 2.2 Objectives
* To perform exploratory data analysis on a real-world job posting dataset.
* To apply Natural Language Processing (NLP) techniques to extract meaning from textual descriptions.
* To train and compare multiple machine learning algorithms.
* To deploy the best-performing model as a functional web application for real-time predictions.

## 3. Dataset Description
The project utilized the Employment Scam Aegean Dataset (EMSCAD), widely known as the "Real / Fake Job Posting Prediction" dataset. 
* **Size**: 17,880 records.
* **Class Imbalance**: Highly imbalanced, with roughly 95% legitimate postings and 5% fraudulent postings.
* **Features**:
  * **Textual**: `title`, `company_profile`, `description`, `requirements`, `benefits`.
  * **Categorical**: `employment_type`, `required_experience`, `required_education`.
  * **Binary**: `telecommuting`, `has_company_logo`, `has_questions`.

## 4. Methodology
### 4.1 Data Preprocessing
* **Missing Value Handling**: Textual columns with missing data were filled with empty strings (`""`), while missing categorical data was labeled as `"Unknown"`.
* **Feature Engineering**: All text features were concatenated into a single `combined_text` column to capture the overall linguistic context of the posting.
* **Vectorization**: Scikit-Learn’s `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) was used to convert the raw text into a numerical matrix, limited to the top 5,000 most significant words.
* **Categorical Encoding**: `OneHotEncoder` was applied to categorical variables to convert them into binary vectors.

### 4.2 Handling Class Imbalance
Because fake jobs represent a very small minority of the dataset, standard accuracy metrics are misleading (a model guessing "Legitimate" every time would achieve 95% accuracy but catch zero scams). 
To counter this, we utilized class-weight balancing (`class_weight='balanced'` and `scale_pos_weight`) during model training, forcing the models to penalize mistakes on the minority "Fraudulent" class more heavily.

### 4.3 Model Selection
We evaluated five different algorithms:
1. **Logistic Regression**: A strong, interpretable baseline.
2. **Multinomial Naive Bayes**: A classic standard for text classification.
3. **Random Forest**: An ensemble method that handles non-linear relationships well.
4. **LinearSVC**: A Support Vector Machine optimized for high-dimensional text data.
5. **XGBoost**: A highly optimized gradient boosting framework known for top-tier performance.

## 5. Model Evaluation and Results
Models were evaluated using Precision, Recall, F1-Score, and the Precision-Recall Area Under the Curve (PR-AUC). 

* **Recall** measures how many of the actual fake jobs were successfully caught.
* **Precision** measures how many of the flagged jobs were actually fake (avoiding false alarms).

| Model | Precision | Recall | F1-Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.5532 | 0.9017 | 0.6857 | 0.8817 |
| Naive Bayes | 0.8852 | 0.3121 | 0.4615 | 0.6534 |
| Random Forest | 0.9903 | 0.5896 | 0.7391 | 0.8930 |
| LinearSVC | 0.7906 | 0.8728 | 0.8297 | 0.9215 |
| **XGBoost** | **0.8683** | **0.8382** | **0.8529** | **0.9294** |

### 5.1 Discussion of Results
* **Naive Bayes** struggled heavily with Recall, catching only 31% of the fake jobs.
* **Logistic Regression** had excellent Recall (90%) but very poor Precision (55%), meaning it flagged far too many legitimate jobs as fake.
* **XGBoost** provided the best overall balance, achieving the highest F1-Score (0.8529). It successfully identified ~84% of all fake jobs while maintaining a high precision of ~87%.

## 6. System Deployment
The final XGBoost model and TF-IDF pipeline were serialized using `joblib` and integrated into a Python **Streamlit** web application designed as an advanced "AI Fraud Detection Engine."

### 6.1 Application Features & UI
* **Cyber Threat Dashboard UI**: The application features a high-tech, glassmorphism aesthetic with neon accents and a dynamic background, mimicking enterprise intelligence tools.
* **Single Analysis & Radar Matrix**: Users can manually input job details or run pre-loaded sample tests. The app instantly generates a "Fraud Probability Score" and renders an interactive **Plotly Radar Threat Matrix** that visually breaks down risk across three axes: Linguistic Risk, Missing Metadata, and Suspicious Requirements.
* **Batch Neural Analysis**: An enterprise-grade feature allowing users to upload a `.csv` dataset of multiple job postings. The engine processes all vectors simultaneously using the XGBoost model, returning a preview DataFrame and generating a downloadable, comprehensive Threat Report CSV.

## 7. Conclusion and Future Work
### 7.1 Conclusion
The project successfully demonstrated that machine learning and NLP can effectively filter out fraudulent job postings. By utilizing TF-IDF and XGBoost with balanced class weights, the system achieved a high degree of reliability, proving that linguistic patterns and missing metadata (like logos or company profiles) are strong indicators of fraud. Furthermore, the deployment of the model into a fully-fledged Cyber Threat Dashboard with Batch Processing capabilities proves that such systems can be scaled for enterprise use.

### 7.2 Future Enhancements
* **Deep Learning / LLMs**: Replacing TF-IDF with advanced embeddings like BERT or fine-tuning a small Large Language Model could capture deeper semantic meanings in scam descriptions.
* **Explainable AI (XAI)**: Integrating SHAP (SHapley Additive exPlanations) into the Streamlit app to highlight the exact words (e.g., "urgent", "wire transfer") that caused a specific posting to be flagged.
* **Web Scraping**: Expanding the system to accept a URL from LinkedIn or Indeed, automatically scraping the text, and returning a risk score.
