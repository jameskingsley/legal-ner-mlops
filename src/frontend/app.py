import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="Legal-NER Classifier", layout="wide", page_icon="⚖️")

st.title("Legal Document Classifier")
st.markdown("""
This dashboard connects to your **FastAPI Backend** to classify legal text into SCOTUS categories.
It pulls the **Champion Model** and **Metadata** directly from ClearML.
""")

API_URL = "http://127.0.0.1:8000"

# --- SIDEBAR: Single Sentence ---
st.sidebar.header("Single Prediction")
user_input = st.sidebar.text_area("Enter legal text to analyze:", height=150, 
                                  placeholder="e.g., The search was conducted without a warrant...")

if st.sidebar.button("Classify Text"):
    if user_input:
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(f"{API_URL}/predict", json={"text": user_input})
                if response.status_code == 200:
                    result = response.json()
                    st.sidebar.success(f"**Category:** {result['category']}")
                    st.sidebar.info(f"**Label ID:** {result['label_id']}")
                else:
                    st.sidebar.error("API Error. Make sure the backend is running.")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")
    else:
        st.sidebar.warning("Please enter some text first.")

# --- MAIN AREA: Batch Upload ---
st.header(" Batch CSV Processing")
st.write("Upload a CSV file with a column named **'text'** to classify multiple documents at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Preview the uploaded file
    df_preview = pd.read_csv(uploaded_file)
    st.write("**File Preview:**")
    st.dataframe(df_preview.head(5))
    
    if st.button("Process Batch Predictions"):
        if 'text' not in df_preview.columns:
            st.error("Missing 'text' column in CSV!")
        else:
            with st.spinner("Processing batch inference..."):
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
                        st.dataframe(results_df)
                        
                        # Download Button
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Classified CSV",
                            data=csv_data,
                            file_name="legal_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Batch Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")