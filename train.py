import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve, confusion_matrix
import joblib

# Configuration
DATA_URL = "https://raw.githubusercontent.com/abbylmm/fake_job_posting/main/data/fake_job_postings.csv"
DATA_FILE = "fake_job_postings.csv"
MODEL_FILE = "fake_job_model.joblib"

def download_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def load_and_preprocess_data():
    df = pd.read_csv(DATA_FILE)
    
    print(f"Dataset shape: {df.shape}")
    
    # 1. Handling Missing Values
    # For text columns, fill missing with empty string
    text_cols = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('')
        
    # For categorical columns, fill with 'Unknown'
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    # Combine text features into a single column
    df['combined_text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']
    
    # Ensure binary columns are integers
    df['telecommuting'] = df['telecommuting'].astype(int)
    df['has_company_logo'] = df['has_company_logo'].astype(int)
    df['has_questions'] = df['has_questions'].astype(int)
    
    # EDA: Save distribution plot
    plt.figure(figsize=(6,4))
    sns.countplot(x='fraudulent', data=df)
    plt.title('Distribution of Fraudulent Jobs')
    plt.savefig('class_distribution.png')
    plt.close()
    
    return df

def build_pipeline(classifier):
    # We will use combined_text for TF-IDF
    # And a few categorical features for OneHotEncoding
    # And numerical (binary) as passthrough
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(stop_words='english', max_features=5000), 'combined_text'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['employment_type', 'required_experience', 'required_education']),
            ('num', 'passthrough', ['telecommuting', 'has_company_logo', 'has_questions'])
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def main():
    download_data()
    df = load_and_preprocess_data()
    
    X = df[['combined_text', 'employment_type', 'required_experience', 'required_education', 'telecommuting', 'has_company_logo', 'has_questions']]
    y = df['fraudulent']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate scale_pos_weight for XGBoost
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(), # MultinomialNB doesn't have class_weight parameter easily available natively for balanced
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "LinearSVC": LinearSVC(class_weight='balanced', random_state=42, dual=False),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # for PR-AUC we need probabilities or decision functions
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = y_pred # fallback
            
        pr_auc = average_precision_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'PR-AUC': pr_auc
        })
        
        print(f"Metrics for {name}:")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | PR-AUC: {pr_auc:.4f}")
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'cm_{name.replace(" ", "_")}.png')
        plt.close()
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    
    print(f"\nSaving Best Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    joblib.dump(best_pipeline, MODEL_FILE)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
