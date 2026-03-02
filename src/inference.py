import joblib
import os
from clearml import Task, Model

def get_latest_champion():
    print(">>> Querying ClearML for champion tasks...")
    
    # 1. Get all tasks in the project
    all_tasks = Task.get_tasks(project_name="Legal-NER-MLOps")
    
    # 2. Filter for 'champion' tag in Python
    champion_tasks = [t for t in all_tasks if 'champion' in t.get_tags()]
    
    if champion_tasks:
        # Sort to get the newest one
        latest_task = sorted(champion_tasks, key=lambda x: x.data.created, reverse=True)[0]
        print(f">>> Found Champion Task: {latest_task.id}")
        
        # 3. Access the model via the task's models attribute (standard dictionary)
        # This bypasses the query_models function entirely
        task_models = latest_task.models
        if task_models and 'output' in task_models and task_models['output']:
            # Get the first output model object
            model_info = task_models['output'][0]
            # Initialize a Model object using the ID found in the task
            model_obj = Model(model_id=model_info.id)
            
            print(f">>> Downloading Model: {model_obj.name} (ID: {model_obj.id})")
            return joblib.load(model_obj.get_local_copy())

    # 4. Fallback to local file
    if os.path.exists("models/best_legal_model.joblib"):
        print(">>> Loading local fallback from models/ directory...")
        return joblib.load("models/best_legal_model.joblib")
    
    raise RuntimeError("Could not find champion model in ClearML or locally.")

def run_inference(text):
    model = get_latest_champion()
    prediction = model.predict([text])[0]
    
    # Mapping for SCOTUS labels
    label_map = {0: "Criminal Procedure", 1: "Civil Rights", 2: "First Amendment"}
    label_name = label_map.get(prediction, f"Label {prediction}")
    
    print(f"\n[Result] Input: {text[:60]}...")
    print(f"[Result] Predicted Category: {label_name} (ID: {prediction})")
    return prediction

if __name__ == "__main__":
    sample_text = "The court must determine if the evidence was obtained through an illegal wiretap."
    run_inference(sample_text)