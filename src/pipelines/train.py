import os
import pandas as pd
import joblib
from ingest import ingestion_pipeline 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from clearml import Task, OutputModel
from prefect import flow, task

# Professional Label Mapping for SCOTUS Dataset
SCOTUS_LABELS = {
    "Criminal Procedure": 0, "Civil Rights": 1, "First Amendment": 2, 
    "Due Process": 3, "Privacy": 4, "Attorneys": 5, "Unions": 6, 
    "Economic Activity": 7, "Judicial Power": 8, "Federalism": 9, 
    "Interstate Relations": 10, "Federal Taxation": 11, "Voting": 12, 
    "Miscellaneous": 13
}

@task(log_prints=True)
def run_model_tournament():
    # 1. Initialize ClearML Task
    task_cl = Task.init(project_name="Legal-NER-MLOps", task_name="Legal-Model-Tournament")
    
    # Store labels in the Task Configuration so they are saved forever with this run
    task_cl.connect(SCOTUS_LABELS, name="labels")
    
    # Check for data
    train_path = "data/raw/train.csv"
    if not os.path.exists(train_path):
        print(">>> Data missing. Running ingestion...")
        ingestion_pipeline()
    
    # Load 5,000 rows
    train_df = pd.read_csv(train_path).sample(n=5000, random_state=42)
    test_df = pd.read_csv("data/raw/test.csv").sample(n=1000, random_state=42)

    # Prepare Models
    models = {
        "SVM": LinearSVC(C=1.0, random_state=42, max_iter=2000),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_test = tfidf.transform(test_df['text'])
    
    best_acc = 0
    winner_name = ""
    winner_model = None

    # The Competition
    for name, clf in models.items():
        clf.fit(X_train, train_df['label'])
        acc = accuracy_score(test_df['label'], clf.predict(X_test))
        task_cl.get_logger().report_single_value(f"accuracy_{name}", acc)
        print(f" {name}: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            winner_name = name
            winner_model = clf

    # Register the Champion
    print(f"\nWINNER: {winner_name}")
    final_pipeline = Pipeline([('tfidf', tfidf), ('clf', winner_model)])
    
    model_path = "models/best_legal_model.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    # Registration
    output_model = OutputModel(task=task_cl)
    # Simple upload - no extra arguments to avoid TypeError
    output_model.update_weights(weights_filename=model_path, auto_delete_file=False)
    
    task_cl.add_tags(["champion", winner_name])
    print(f">>> Model registered and uploaded to ClearML.")

@flow(name="Tournament-Flow")
def run_pipeline():
    run_model_tournament()

if __name__ == "__main__":
    run_pipeline()