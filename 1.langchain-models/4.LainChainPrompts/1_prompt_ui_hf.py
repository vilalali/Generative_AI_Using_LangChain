import streamlit as st
from langchain_community.llms import HuggingFaceHub  # Updated import
from huggingface_hub import InferenceClient  # New API

# Initialize Hugging Face Inference Client (Replaces deprecated HuggingFaceHub)
#client = InferenceClient(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1")

# Streamlit UI
st.title("LangChain Hugging Face Chatbot")
user_input = st.text_area("Enter your prompt:", "Once upon a time, i am in love...")

if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Use the new InferenceClient API for text generation
                response = client.text_generation(user_input)
                st.write("### Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
