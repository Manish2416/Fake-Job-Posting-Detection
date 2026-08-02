import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fake Job Posting Detection\n",
    "\n",
    "This notebook covers the end-to-end machine learning pipeline for detecting fraudulent job postings. It includes Data Loading, Exploratory Data Analysis, Preprocessing, Feature Engineering, and Model Training/Evaluation across multiple algorithms."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import urllib.request\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import LinearSVC\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load the Dataset\n",
    "We load the `fake_job_postings.csv` dataset. The dataset contains both textual data (e.g., job descriptions) and categorical metadata."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_FILE = \"fake_job_postings.csv\"\n",
    "df = pd.read_csv(DATA_FILE)\n",
    "print(f\"Dataset shape: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis\n",
    "Let's visualize the target class distribution (`fraudulent` column) to check for class imbalance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6,4))\n",
    "sns.countplot(x='fraudulent', data=df)\n",
    "plt.title('Distribution of Fraudulent Jobs')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Data Preprocessing\n",
    "We need to handle missing values:\n",
    "- **Text Columns**: Fill with empty strings.\n",
    "- **Categorical Columns**: Fill with 'Unknown'.\n",
    "\n",
    "We will also combine all relevant text fields into a single `combined_text` column to process them with TF-IDF."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Handle missing values in text columns\n",
    "text_cols = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']\n",
    "for col in text_cols:\n",
    "    df[col] = df[col].fillna('')\n",
    "    \n",
    "# Handle missing values in categorical columns\n",
    "cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']\n",
    "for col in cat_cols:\n",
    "    df[col] = df[col].fillna('Unknown')\n",
    "\n",
    "# Combine text features into a single column\n",
    "df['combined_text'] = df['title'] + \" \" + df['company_profile'] + \" \" + df['description'] + \" \" + df['requirements'] + \" \" + df['benefits']\n",
    "\n",
    "# Ensure binary columns are integers\n",
    "df['telecommuting'] = df['telecommuting'].astype(int)\n",
    "df['has_company_logo'] = df['has_company_logo'].astype(int)\n",
    "df['has_questions'] = df['has_questions'].astype(int)\n",
    "\n",
    "print(\"Preprocessing Complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Feature Engineering and Pipeline Setup\n",
    "We use `ColumnTransformer` to handle different types of features simultaneously:\n",
    "- **TfidfVectorizer** for unstructured text (`combined_text`)\n",
    "- **OneHotEncoder** for categorical variables\n",
    "- **passthrough** for binary indicators"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_pipeline(classifier):\n",
    "    preprocessor = ColumnTransformer(\n",
    "        transformers=[\n",
    "            ('text', TfidfVectorizer(stop_words='english', max_features=5000), 'combined_text'),\n",
    "            ('cat', OneHotEncoder(handle_unknown='ignore'), ['employment_type', 'required_experience', 'required_education']),\n",
    "            ('num', 'passthrough', ['telecommuting', 'has_company_logo', 'has_questions'])\n",
    "        ]\n",
    "    )\n",
    "    \n",
    "    pipeline = Pipeline(steps=[\n",
    "        ('preprocessor', preprocessor),\n",
    "        ('classifier', classifier)\n",
    "    ])\n",
    "    \n",
    "    return pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Training & Evaluation\n",
    "We will evaluate several models, ensuring we handle the class imbalance using parameters like `class_weight='balanced'` and `scale_pos_weight`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[['combined_text', 'employment_type', 'required_experience', 'required_education', 'telecommuting', 'has_company_logo', 'has_questions']]\n",
    "y = df['fraudulent']\n",
    "\n",
    "# 80-20 Train-Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# Calculate scale_pos_weight for XGBoost\n",
    "neg_count = sum(y_train == 0)\n",
    "pos_count = sum(y_train == 1)\n",
    "scale_pos_weight = neg_count / pos_count\n",
    "\n",
    "models = {\n",
    "    \"Logistic Regression\": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),\n",
    "    \"Naive Bayes\": MultinomialNB(),\n",
    "    \"Random Forest\": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),\n",
    "    \"LinearSVC\": LinearSVC(class_weight='balanced', random_state=42, dual=False),\n",
    "    \"XGBoost\": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')\n",
    "}\n",
    "\n",
    "best_f1 = 0\n",
    "best_model_name = \"\"\n",
    "best_pipeline = None\n",
    "results = []\n",
    "\n",
    "for name, model in models.items():\n",
    "    print(f\"\\nTraining {name}...\")\n",
    "    pipeline = build_pipeline(model)\n",
    "    pipeline.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred = pipeline.predict(X_test)\n",
    "    \n",
    "    if hasattr(pipeline, \"predict_proba\"):\n",
    "        y_prob = pipeline.predict_proba(X_test)[:, 1]\n",
    "    elif hasattr(pipeline, \"decision_function\"):\n",
    "        y_prob = pipeline.decision_function(X_test)\n",
    "    else:\n",
    "        y_prob = y_pred\n",
    "        \n",
    "    pr_auc = average_precision_score(y_test, y_prob)\n",
    "    precision = precision_score(y_test, y_pred)\n",
    "    recall = recall_score(y_test, y_pred)\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    \n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Precision': precision,\n",
    "        'Recall': recall,\n",
    "        'F1-Score': f1,\n",
    "        'PR-AUC': pr_auc\n",
    "    })\n",
    "    \n",
    "    print(f\"Metrics for {name}:\")\n",
    "    print(f\"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | PR-AUC: {pr_auc:.4f}\")\n",
    "    \n",
    "    # Print Confusion Matrix\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    plt.figure(figsize=(4,3))\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title(f'Confusion Matrix: {name}')\n",
    "    plt.ylabel('Actual')\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.show()\n",
    "    \n",
    "    if f1 > best_f1:\n",
    "        best_f1 = f1\n",
    "        best_model_name = name\n",
    "        best_pipeline = pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Results Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df = pd.DataFrame(results)\n",
    "results_df.sort_values(by='F1-Score', ascending=False, inplace=True)\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Save the Best Model\n",
    "We save the top performing model so it can be integrated into the Streamlit Web App."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "MODEL_FILE = \"fake_job_model.joblib\"\n",
    "print(f\"Saving Best Model: {best_model_name} with F1-Score: {best_f1:.4f}\")\n",
    "joblib.dump(best_pipeline, MODEL_FILE)\n",
    "print(\"Model saved successfully.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('c:/Users/Manish Sahu/.gemini/antigravity/scratch/Fake-Job-Posting-Detection/Fake_Job_Posting_Detection.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
