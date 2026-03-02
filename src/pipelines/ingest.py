import os
from prefect import flow, task
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

@task(retries=2, description="Fetch Legal NER data from Hugging Face")
def fetch_legal_data():
    # Using the 'le_ner' dataset (Brazilian Legal) or 'conll2003' for generic legal testing
    dataset = load_dataset("coastalcph/lex_glue", "scotus") 
    print(" Dataset downloaded successfully")
    return dataset

@task(description="Save raw data to DVC-tracked folder")
def save_to_local(dataset):
    raw_path = "data/raw"
    os.makedirs(raw_path, exist_ok=True)
    
    # Convert to DataFrame for easier inspection/versioning
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    train_df.to_csv(f"{raw_path}/train.csv", index=False)
    test_df.to_csv(f"{raw_path}/test.csv", index=False)
    print(f"Files saved to {raw_path}")

@flow(name="Data Ingestion Flow")
def ingestion_pipeline():
    ds = fetch_legal_data()
    save_to_local(ds)

if __name__ == "__main__":
    ingestion_pipeline()