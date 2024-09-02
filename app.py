import os
import dotenv
import streamlit as st
from typing import Generator
from groq import Groq

dotenv.load_dotenv()

st.set_page_config(page_icon="💬", layout="wide", page_title="TARS ChatBot")

def display_image(image_path: str):
    """Displays an image centered on the page."""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col2:
        st.image(image_path, width=220)

display_image("TARS.png")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial message from TARS
    st.session_state.messages.append({"role": "assistant", "content": "Hello there! I am TARS (Tactical Assistance and Response System), a bootleg version of the TARS from Interstellar. How can I assist you today?"})

# Define model details
model_option = "llama3-8b-8192"

models = {
    "llama3-8b-8192": {
        "name": "Llama3-8b-Instruct-v0.1",
        "tokens": 8192,
        "developer": "Meta AI",
    },
}

max_tokens_range = models[model_option]["tokens"]

def reset_context_length():
    st.session_state.messages = []
    # Optionally, re-add the initial message when resetting the chat
    st.session_state.messages.append({"role": "assistant", "content": "Hello there! I am TARS (Tactical Assistance and Response System), a bootleg version of the TARS from Interstellar. How can I assist you today?"})

# Add custom CSS for centering chat messages
st.markdown("""
    <style>
        .stChatMessage {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 10px;
        }
        .stChatMessage__avatar {
            margin-right: 10px;
        }
        .stChatMessage__content {
            display: inline-block;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Create a column layout
col1, _, _ = st.columns([1, 0.1, 0.1])

with col1:
    # Add a reset context button
    reset_button = st.button(
        "New Chat", on_click=reset_context_length, key="reset_button"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] != "system":  # Skip displaying system messages
        avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("What's on your mind?!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑🏼‍🚀"):
        st.markdown(prompt)

    # Update the system message to instruct the model to respond in English
    st.session_state.messages.insert(0, {"role": "system", "content": "Please respond in English."})

    # Check if the user is asking for the bot's name
    if "name" in prompt.lower() or "who are you" in prompt.lower() or "what are you" in prompt.lower():
        response = "Hello there! I am TARS (Tactical Assistance and Response System), a bootleg version of the TARS from Interstellar. How can I assist you today?"
    else:
        # Fetch response from Groq API
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                max_tokens=max_tokens_range,
                stream=True,
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            response = full_response
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            response = combined_response

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
