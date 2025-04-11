import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Load Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Configure global settings
Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=GEMINI_API_KEY)
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900


# Save uploaded file temporarily and load
def load_data_from_upload(uploaded_file):
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = SimpleDirectoryReader(input_files=[temp_path])
            documents = loader.load_data()
            return documents
    else:
        loader = SimpleDirectoryReader("Data")
        documents = loader.load_data()
        return documents


# Load Gemini-Pro model
def load_model():
    model = Gemini(models='gemini-pro', api_key=GEMINI_API_KEY)
    return model


# Generate index using global Settings
def download_gemini_embedding(document):
    index = VectorStoreIndex.from_documents(document, service_context=Settings)
    index.storage_context.persist()
    query_engine = index.as_query_engine()
    return query_engine


# Streamlit UI
def main():
    st.set_page_config(page_title="QA with Documents")
    st.header("📄 QA with Documents")

    st.markdown("Upload a document (PDF or text), then ask a question about its content!")

    # Session state to store query engine
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None

    uploaded_file = st.file_uploader("Upload your document")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing and embedding the document..."):
            documents = load_data_from_upload(uploaded_file)
            load_model()  
            st.session_state.query_engine = download_gemini_embedding(documents)
            st.success("✅ Document embedded successfully. You can now ask questions.")

    # Ask question only after processing
    if st.session_state.query_engine:
        user_question = st.text_input("💬 Ask your question")
        if user_question:
            with st.spinner("Generating answer..."):
                response = st.session_state.query_engine.query(user_question)
                st.success("✅ Answer:")
                st.write(response.response)

if __name__ == "__main__":
    main()