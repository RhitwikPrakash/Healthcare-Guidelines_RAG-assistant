# Healthcare Guidelines RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant that answers questions from uploaded healthcare guideline PDFs using retrieved evidence and page-level references.

## Features

- Upload and process multiple medical PDFs
- Extract text using PyMuPDF
- Build a TF-IDF retrieval index with scikit-learn
- Retrieve relevant evidence using cosine similarity
- Generate document-grounded answers with Google Gemini
- Use Groq as a fallback LLM provider
- Fast, Smart Auto, and Deep Research answer modes
- Display references at the end of answers
- Inspect retrieved evidence and source pages
- Login authentication using configured credentials
- Session-based chat history, switching and renaming
- Export conversations as Markdown
- API request pacing and rate-limit handling

## Technology Stack

- Frontend and application: Python, Streamlit
- PDF extraction: PyMuPDF
- Retrieval: TF-IDF, cosine similarity, scikit-learn
- Numerical operations: NumPy
- Primary LLM: Google Gemini
- Fallback LLM: Groq
- API communication: Requests
- Authentication: Streamlit Secrets and session state
- Deployment: Streamlit Community Cloud

## RAG Architecture

PDF Upload
    |
PDF Text Extraction
    |
Text Chunking
    |
TF-IDF Indexing
    |
Question Processing
    |
Cosine Similarity Retrieval
    |
Evidence Selection
    |
Grounded Prompt Construction
    |
Gemini / Groq
    |
Answer + References + Evidence Cards

## Project Structure

cloud_lite/
    app.py
    auth_utils.py
    llm_client.py
    pdf_utils.py
    rag_core.py
    request_gate.py
    requirements.txt
    .streamlit/
        config.toml
        secrets.toml.example

## Local Setup

1. Clone the repository.
2. Open the cloud_lite directory.
3. Install dependencies:

   python -m pip install -r requirements.txt

4. Create a private .streamlit/secrets.toml file using
   secrets.toml.example as a template.
5. Configure authentication and LLM API credentials.
6. Start the application:

   python -m streamlit run app.py

## Streamlit Cloud Deployment

Main application file:

cloud_lite/app.py

Install dependencies from:

cloud_lite/requirements.txt

Configure AUTH_USERNAME, AUTH_EMAIL, AUTH_PASSWORD,
GOOGLE_API_KEY and GROQ_API_KEY in Streamlit Cloud Secrets.

Never commit actual passwords or API keys.

## Limitations

- Chat history is retained only during the current session.
- Retrieval is primarily lexical and may miss semantic matches.
- Answers depend on the uploaded evidence and LLM output.
- This application is not a medical diagnostic or prescribing tool.
- Confidential medical data should not be uploaded without
  appropriate authorization and provider privacy review.

## License

MIT License.
