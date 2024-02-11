import streamlit as st
import pdfplumber
import os
from langchain_openai import ChatOpenAI

# Get the OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("AI PDF Analyzer")
st.write("by YeppRK")

# Instantiate the fine-tuned model with the OpenAI API key
fine_tuned_model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    temperature=0.5,
    model_name="gpt-4-0125-preview"
)

# Streamlit UI
uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")
llm_string = st.text_input("Instructions/Question")
button_clicked = st.button("Process")

if button_clicked:
    if uploaded_file is not None and llm_string.strip() != "":
        # Read the content of the uploaded PDF file
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            st.stop()

        # Pass the PDF text and user input to the fine-tuned model
        inputs = [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes a given PDF and answers questions based on it."
            },
            {
                "role": "user",
                "content": f"Text: {pdf_text}\nInstruction: {llm_string}"
            }
        ]
        try:
            result = fine_tuned_model.invoke(inputs, config=None)
            st.write("Result:")
            st.write(result.content)
        except Exception as e:
            st.error(f"Error processing with fine-tuned model: {e}")
    else:
        st.warning("Upload a PDF file and enter instructions before processing.")