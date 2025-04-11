# Document Q&A with Google Gemini

A Streamlit-based Retrieval-Augmented Generation (RAG) application that uses Google's Gemini AI models and LlamaIndex to answer questions about uploaded documents.
This project implements a document question-answering system powered by Google's Generative AI technology. It allows users to upload documents and get AI-generated answers based on the document content.

### Key GenAI Components

- **Google Gemini Models**: Uses Gemini-2.0-flash for answer generation and text-embedding-004 for document embeddings
- **RAG Pipeline**: Implements Retrieval-Augmented Generation to enhance answer quality with document context
- **LlamaIndex Integration**: Leverages LlamaIndex for document processing, chunking, and vector retrieval

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the app: `streamlit run app.py`
2. Upload a PDF or text document
3. Click "Process Document" 
4. Ask questions about your document

