## TARS ChatBot

TARS is a conversational chatbot built with Streamlit and Groq, inspired by the TARS robot from *Interstellar*. This chatbot offers users a friendly and engaging experience with advanced features for document-based interactions and Retrieval-Augmented Generation (RAG).

## Website link 🔗
[https://tars-the-quantum-bot.streamlit.app/](https://tars-the-quantum-bot.streamlit.app/)

## Features

- **Interactive Chat Interface**: Engage with TARS through a user-friendly chat layout.
- **Real-time Responses**: Utilizes GROQ API with the `llama-3.1-70b-versatile` model to fetch and display responses instantly.
- **Document Upload and Processing**: Upload PDFs, DOCX, and TXT files to enhance the chatbot’s responses using document-based retrieval.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with language generation to provide more accurate and contextually relevant answers.
- **Customizable Model**: Currently uses the `llama-3.1-70b-versatile` model for generating responses.
- **Reset Chat**: Option to start a new chat session.

## Retrieval-Augmented Generation (RAG)

RAG is a technique that improves conversational AI by integrating document retrieval with generative responses. Here's how it works in TARS:

- **Document Upload and Processing**: Users can upload documents, which are then processed and split into chunks using LangChain's document loaders and text splitters.
- **Vector Store**: The processed document chunks are stored in an in-memory vector database (Chroma), allowing for efficient retrieval of relevant information.
- **Conversational Retrieval Chain**: The RAG model combines the capabilities of document retrieval and generative response. When a user asks a question, TARS retrieves relevant document chunks and generates responses based on this information, enhancing the accuracy and relevance of the answers.

## Setup

### Prerequisites

- Python 3.7+
- Streamlit
- Groq API
- LangChain and related libraries

### Installation

1. **Install Required Packages**:

    ```bash
    pip install streamlit groq langchain langchain_community
    ```

2. **Set Up Environment Variables**:

    Create a `.env` file in the project directory with your Groq API key:

    ```env
    GROQ_API_KEY=your_groq_api_key
    ```

### Running the Application

1. **Start the Streamlit App**:

    ```bash
    streamlit run app.py
    ```

2. **Access the App**:

    Open a web browser and navigate to `http://localhost:8501` to interact with TARS.

## Customization

- **Change Model**: Modify the `model_option` variable in `app.py` to use different models.
- **Update Greeting**: Edit the initial message in the `st.session_state.messages` list for a different welcome message.

## Troubleshooting

- Ensure the `.env` file contains the correct Groq API key.
- Check your internet connection if you experience issues with responses.
- Ensure your document files are in PDF, DOCX, or TXT formats.

## Contact

For questions or suggestions, please reach out to [mokakrishna212@gmail.com](mailto:mokakrishna212@gmail.com).
