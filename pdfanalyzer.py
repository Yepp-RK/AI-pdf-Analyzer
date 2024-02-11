import streamlit as st
import pdfplumber
from langchain_openai import ChatOpenAI
import os


st.title("AI PDF Analyzer")
st.subheader("by YeppRK")

api_key = os.getenv('OPENAI_API_KEY')


llm = ChatOpenAI(api_key=api_key, temperature=0.5)
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

        # Pass the PDF text and user input to the ChatOpenAI model
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
            result = llm.invoke(inputs, config=None)
            st.write("Result:")
            st.write(result.content)
        except Exception as e:
            st.error(f"Error processing with langchain_openai: {e}")
    else:
        st.warning("Upload a PDF file and enter instructions before processing.")

