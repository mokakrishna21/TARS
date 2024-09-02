## TARS ChatBot

TARS is a conversational chatbot built with Streamlit and Groq. Inspired by the TARS robot from Interstellar, this chatbot provides a friendly and engaging experience for users.

## Features

- **Interactive Chat Interface**: Engage with TARS through a user-friendly chat layout.
- **Real-time Responses**: Utilizes GROQ API to fetch and display responses instantly.
- **Customizable Model**: Uses the Llama3-8b-8192 model for generating responses.
- **Reset Chat**: Option to start a new chat session.

## Setup

### Prerequisites

- Python 3.7+
- Streamlit
- Groq API

### Installation

1. **Install Required Packages**:

    ```bash
    pip install streamlit groq
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

## Contact

For questions or suggestions, please reach out to [mokakrishna212@gmail.com].