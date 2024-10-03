# RAG-Doc-Assistant

# Document RAG Chatbot

## Overview

The Document RAG Chatbot is an intelligent assistant designed to interact with various document types, including PDF, CSV, JSON, DOCX, XLSX, and PPTX files. Using Retrieval-Augmented Generation (RAG) techniques, this chatbot provides accurate responses to user queries based on the content of the uploaded documents.

## Features

- **Document Upload**: Supports multiple file formats (PDF, CSV, JSON, DOCX, XLSX, PPTX).
- **Contextual Querying**: Leverages embeddings and a FAISS index for efficient searching and retrieval of relevant document content.
- **Azure OpenAI Integration**: Utilizes the Azure OpenAI API for generating responses based on the retrieved context.
- **User-Friendly Interface**: Built using Streamlit for a seamless user experience.

## Requirements

- Python 3.8 or higher
- Required libraries (can be installed via `requirements.txt`):
  - `faiss-cpu`
  - `numpy`
  - `pandas`
  - `langchain`
  - `sentence-transformers`
  - `streamlit`
  - `python-dotenv`
  - `python-docx`
  - `python-pptx`
  - `openai`

## Installation

1. Clone the repository:
   git clone <repository-url>
   cd <repository-name>
   
2. Install the required libraries:
pip install -r requirements.txt

3. Set up your environment variables by creating a .env file in the root directory. This file should contain your Azure OpenAI credentials:
AZURE_OPENAI_API_VERSION=<your-azure-openai-api-version>
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
OPENAI_API_BASE=<your-openai-api-base-url>
- Replace <your-azure-openai-api-version>, <your-azure-openai-api-key>, and <your-openai-api-base-url> with your specific values.

4. Usage:
- Run the Streamlit app:
   -- streamlit run app.py
- Open your browser and navigate to http://localhost:8501.
- Upload your documents using the file uploader.
- Enter your query in the input box and receive responses based on the uploaded documents.



License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Langchain for simplifying document loading and processing.
FAISS for enabling fast similarity search.
