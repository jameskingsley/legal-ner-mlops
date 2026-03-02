from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from clearml import Task, Model
import joblib
import uvicorn
import os
import pandas as pd
import io

app = FastAPI(title="Legal-NER Classification Service")

# Hardcoded fallback to ensure the API never crashes on startup
SCOTUS_LABELS = {
    0: "Criminal Procedure", 1: "Civil Rights", 2: "First Amendment", 
    3: "Due Process", 4: "Privacy", 5: "Attorneys", 6: "Unions", 
    7: "Economic Activity", 8: "Judicial Power", 9: "Federalism", 
    10: "Interstate Relations", 11: "Federal Taxation", 12: "Voting", 
    13: "Miscellaneous"
}

def load_production_model():
    print(">>> Fetching Champion from ClearML...")
    try:
        # Find the latest champion task
        all_tasks = Task.get_tasks(project_name="Legal-NER-MLOps")
        champion_tasks = [t for t in all_tasks if 'champion' in t.get_tags()]
        
        if not champion_tasks:
            raise ValueError("No tasks with 'champion' tag found.")
            
        latest_task = sorted(champion_tasks, key=lambda x: x.data.created, reverse=True)[0]
        print(f">>> Found Task: {latest_task.id}")

        # Advanced Label Detection
        id_to_name = None
        
        # Check Configuration Objects first
        labels_cfg = latest_task.get_configuration_object("labels")
        
        # If not found, check Hyperparameters (common for task.connect)
        if not labels_cfg:
            params = latest_task.get_parameters_as_dict()
            labels_cfg = params.get('labels')

        if labels_cfg:
            print(">>> Success! Found labels in ClearML metadata.")
            # ClearML stores everything as strings; convert keys to int for the model
            id_to_name = {int(v): k for k, v in labels_cfg.items()}
        else:
            print(">>> Labels not found in Task metadata. Using local fallback.")
            id_to_name = SCOTUS_LABELS

        # 3. Get Model Weights
        task_models = latest_task.models
        if not task_models.get('output'):
            raise ValueError("Champion task has no output model.")
            
        model_id = task_models['output'][0].id
        model_path = Model(model_id=model_id).get_local_copy()
        
        return joblib.load(model_path), id_to_name

    except Exception as e:
        print(f">>> Error loading from ClearML: {e}")
        print(">>> CRITICAL: Falling back to local model file...")
        # Emergency local fallback 
        fallback_path = "models/best_legal_model.joblib"
        if os.path.exists(fallback_path):
            return joblib.load(fallback_path), SCOTUS_LABELS
        else:
            raise RuntimeError("No model found in ClearML or local /models directory.")

# Load once on startup to keep inference fast
clf_model, label_map = load_production_model()

class LegalQuery(BaseModel):
    text: str

# Endpoint 1: Single Prediction 
@app.post("/predict")
async def predict(query: LegalQuery):
    pred_id = int(clf_model.predict([query.text])[0])
    category = label_map.get(pred_id, f"Category {pred_id}")
    
    return {
        "text_preview": query.text[:60] + "...",
        "category": category,
        "label_id": pred_id
    }

# Endpoint 2: Batch Prediction (CSV) 
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV content
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")

        # Run Batch Inference
        predictions = clf_model.predict(df['text'].fillna("").tolist())
        
        # Add results to the dataframe
        df['label_id'] = predictions
        df['category'] = df['label_id'].map(label_map)
        
        # Return the result as a list of dictionaries
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)