import streamlit as st
import requests
import pandas as pd
import io

# Page Config
st.set_page_config(page_title="Legal-NER Classifier", layout="wide", page_icon="")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0e1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Update this to your ACTUAL Render URL
API_URL = "https://legal-ner-mlops.onrender.com"

st.title("SCOTUS Case Classifier")
st.markdown(f"""
This dashboard connects to your **Production API on Render**.
It utilizes the **Champion Model** (SVM) and labels pulled dynamically from **ClearML**.
Currently connected to: `{API_URL}`
""")

# --- CONNECTION CHECK ---
def check_api():
    try:
        # We use the /docs or a simple health ping if you added one
        # For now, a simple GET to the root to see if it responds
        response = requests.get(API_URL, timeout=5)
        return True
    except:
        return False

# Sidebar Connection Status
with st.sidebar:
    st.header("Backend Status")
    if check_api():
        st.success("● API Online")
    else:
        st.error("○ API Offline (Waking up...)")
        st.caption("Note: Render's free tier may take 30-60s to boot up if it has been inactive.")
    
    st.divider()
    st.header("Single Prediction")
    user_input = st.text_area("Enter legal text to analyze:", height=150, 
                             placeholder="e.g., The search was conducted without a warrant...")

    if st.button("Classify Text"):
        if user_input:
            try:
                with st.spinner("Analyzing..."):
                    response = requests.post(f"{API_URL}/predict", json={"text": user_input})
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"**Category:** {result['category']}")
                        st.info(f"**Label ID:** {result['label_id']}")
                    else:
                        st.error("API Error. The service might still be initializing.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
        else:
            st.warning("Please enter some text first.")

# MAIN AREA: Batch Upload 
st.header("Batch CSV Processing")
st.write("Upload a CSV file with a column named **'text'** to classify multiple documents at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Preview the uploaded file
    df_preview = pd.read_csv(uploaded_file)
    st.write("**File Preview:**")
    st.dataframe(df_preview.head(5), use_container_width=True)
    
    if st.button("Process Batch Predictions"):
        if 'text' not in df_preview.columns:
            st.error("Missing 'text' column in CSV!")
        else:
            with st.spinner("Processing batch inference on Render..."):
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                
                try:
                    response = requests.post(f"{API_URL}/predict_batch", files=files)
                    if response.status_code == 200:
                        results_df = pd.DataFrame(response.json())
                        
                        st.divider()
                        st.success(f"Done! Processed {len(results_df)} rows.")
                        
                        # Display Results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download Button
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=" Download Classified CSV",
                            data=csv_data,
                            file_name="legal_predictions_processed.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Batch Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# FOOTER
st.divider()
st.caption("Legal-NER MLOps Project | Built with FastAPI, ClearML, and Streamlit")