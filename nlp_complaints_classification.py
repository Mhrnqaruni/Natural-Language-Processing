# -*- coding: utf-8 -*-
"""
NLP Complaint Classification
============================

This script loads consumer complaint data, preprocesses the text,
generates vector embeddings with a pretrained Word2Vec model, handles
class imbalance with SMOTE, then trains and compares multiple classifiers.
Finally, it plots accuracy and F1-scores, and displays a confusion matrix
for the best-performing classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

# Attempt to download the 'stopwords' if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import gensim.downloader as api

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE


def main():

    ########################################################
    # 1. Load and Clean the Dataset
    ########################################################
    print("Loading dataset...")
    df = pd.read_csv("complaints_processed.csv")

    # Drop any extra columns that might come from certain CSV exports
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Replace empty strings in 'narrative' with NaN, then drop missing
    df['narrative'] = df['narrative'].replace('', np.nan)
    df = df.dropna(subset=['narrative'])

    # OPTIONAL: Reduce data size for speed
    df = df.iloc[:5000].copy()

    ########################################################
    # 2. Load Pretrained Word2Vec Model
    ########################################################
    print("Loading Word2Vec model (Google News 300)...")
    word2vec = api.load('word2vec-google-news-300')  # Large model (~1.6GB)

    ########################################################
    # 3. Preprocess and Vectorize Narratives
    ########################################################
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))

    def preprocess_vectorize(text):
        words = tokenizer.tokenize(text)
        words_lower = [w.lower() for w in words]
        tokens = []
        for token in words_lower:
            if token not in stop_words and token in word2vec.key_to_index:
                tokens.append(token)

        if tokens:
            return word2vec.get_mean_vector(tokens)
        else:
            return np.nan

    # Create 'vector' column
    df['vector'] = df['narrative'].apply(preprocess_vectorize)
    # Drop any rows where 'vector' ended up being NaN
    df = df.dropna(subset=['vector'])

    ########################################################
    # 4. Encode Labels
    ########################################################
    # Map product categories to numeric labels. Adjust if your dataset differs.
    product_mapping = {
        'credit_card': 0,
        'retail_banking': 1,
        'credit_reporting': 2,
        'mortgages_and_loans': 3,
        'debt_collection': 4
    }

    df['product_num'] = df['product'].map(product_mapping)
    # Drop rows where 'product' was unmapped (i.e., not in product_mapping)
    df = df.dropna(subset=['product_num'])

    # Features (X) and labels (y)
    X = np.stack(df['vector'].values)
    y = df['product_num'].values.astype(int)

    ########################################################
    # 5. Handle Class Imbalance with SMOTE
    ########################################################
    print("Performing SMOTE oversampling for class imbalance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    ########################################################
    # 6. Split into Training and Test Sets
    ########################################################
    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res
    )

    ########################################################
    # 7. Train and Compare Classifiers
    ########################################################
    classifiers = {
        "RBF SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Neural Net": MLPClassifier(max_iter=300),
        "Voting": VotingClassifier(estimators=[
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier()),
            ('mlp', MLPClassifier(max_iter=300))
        ], voting='soft')
    }

    results = []
    print("Training classifiers and evaluating...")
    for name, clf in classifiers.items():
        # Pipeline: Scale data => classifier
        pipeline = make_pipeline(StandardScaler(), clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            'classifier': name,
            'accuracy': acc,
            'f1_score': f1,
            'pipeline': pipeline
        })
        print(f"{name}: accuracy={acc:.4f}, f1={f1:.4f}")

    ########################################################
    # 8. Plot Comparison of Classifiers
    ########################################################
    results_df = pd.DataFrame(results)
    # Sort by F1-score in descending order
    results_df.sort_values(by='f1_score', ascending=False, inplace=True)

    classifiers_list = results_df['classifier'].values
    accuracy_list = results_df['accuracy'].values
    f1_list = results_df['f1_score'].values

    # Plot Accuracy
    plt.figure()
    plt.bar(classifiers_list, accuracy_list)
    plt.title("Accuracy Comparison of Classifiers")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")  # Optional: Save figure
    plt.show()

    # Plot F1-Score
    plt.figure()
    plt.bar(classifiers_list, f1_list)
    plt.title("F1-Score Comparison of Classifiers")
    plt.xlabel("Classifier")
    plt.ylabel("Weighted F1")
    plt.tight_layout()
    plt.savefig("f1_comparison.png")  # Optional: Save figure
    plt.show()

    ########################################################
    # 9. Confusion Matrix for the Best Classifier
    ########################################################
    best_row = results_df.iloc[0]
    best_name = best_row['classifier']
    best_pipeline = best_row['pipeline']

    print(f"\nBest classifier by F1-score: {best_name}")

    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=product_mapping.keys())
    disp.plot()
    plt.title(f"Confusion Matrix: {best_name}")
    plt.savefig("best_confusion_matrix.png")  # Optional: Save figure
    plt.show()


if __name__ == "__main__":
    main()
