# Legal-NER MLOps: Enterprise-Grade SCOTUS Classification Pipeline
######  Project Overview
This repository implements a robust, end-to-end MLOps ecosystem for classifying United States Supreme Court (SCOTUS) legal documents into 14 thematic categories. Unlike traditional "notebook-based" ML, this project treats the model as a living service, incorporating automated orchestration, experiment tracking, and elastic deployment.

###### Technical Architecture
The system is designed with a Decoupled Serving Architecture. The API layer is agnostic of the model's internal parameters, dynamically resolving the "Champion" model via a centralized registry at runtime.

###### Core Stack
* Orchestration (Prefect): Manages the DAG (Directed Acyclic Graph) for data ingestion, preprocessing, and the "Model Tournament."

* Data Version Control (DVC): Ensures 100% data lineage and reproducibility by versioning large datasets outside of Git.

* Experiment Management (ClearML): Functions as the Model Registry and Metadata Store. Tracks hyperparameter tuning, F1-scores, and artifacts.

* Inference Layer (FastAPI): An asynchronous REST API optimized for both high-concurrency single-point inference and vectorized batch processing.

* UI Layer (Streamlit): A reactive frontend for domain experts to interact with the model without technical overhead.

###### Key Engineering Features
* Champion-Tag Deployment: Automated model promotion logic. The API queries the ClearML API for the specific champion tag, allowing for seamless updates without redeploying code.

* Vectorized Batch Inference: The /predict_batch endpoint leverages pandas and scikit-learn vectorization to process thousands of legal records with minimal latency.

* Hybrid Metadata Loading: Implements an advanced lookup logic that syncs label mappings directly from the ClearML task configuration, ensuring the UI and Model are always in vertical alignment.

* Resiliency & Fallback: Built-in "Graceful Degradation"—if the cloud registry is unavailable, the system automatically initializes using localized "Last Known Good" (LKG) weights.

###### Repository Structure

* .dvc/               
* config/             
* data/               
* src/
   * api/            
  * pipeline/          
  * frontend.py    
* render.yaml         
* requirements.txt    

###### Installation & Setup
*  Environment Initialization

* git clone https://github.com/username/legal-ner-mlops.git
* cd legal-ner-mlops
* python -m venv venv
* source venv/bin/activate  # On Windows: venv\Scripts\activate
* pip install -r requirements.txt
###### Data & Experiment Setup
Ensure DVC is configured with your remote (S3/GDrive/Azure) and ClearML credentials are set in config/clearml.conf.

* dvc pull
* python src/flows/train.py  # Executes the Model Tournament
###### Local Deployment

* Start the Backend (Port 8000)
* python src/api/main.py
* Start the Frontend (Port 8501)
* streamlit run src/frontend/app.py
###### Continuous Deployment (CD)
* Backend: Automatically deployed to Render via Blueprint (render.yaml).
* Inference Secret Management: ClearML API keys are injected via environment variables to maintain security.
* Frontend: Hosted on Streamlit Cloud, fetching predictions from the production FastAPI URL.

## Maintained by: James Kingsley