import os
import sys
import tempfile
import random
import streamlit as st
import json
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
import dotenv
from streamlit_mic_recorder import speech_to_text

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION to python as a workaround
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Use pysqlite3 as a drop-in replacement for sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

dotenv.load_dotenv()

# Initialize Groq API Client
groq_api_key = os.getenv("GROQ_API_KEY")
client = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.5,
    groq_api_key=groq_api_key
)

# UI Configurations
st.set_page_config(page_icon="🌌", layout="wide", page_title="TARS")

# Display Image
def display_image(image_path: str):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        st.image(image_path, width=220)

display_image("TARS.png")

# Voice Recording Function
def record_voice(language="en"):
    state = st.session_state
    if "text_received" not in state:
        state.text_received = []
    
    text = speech_to_text(
        start_prompt="🎤 Click and speak",
        stop_prompt="⏹️ Stop recording",
        language=language,
        use_container_width=True,
        just_once=True,
        key=f"recorder_{language}"
    )
    
    if text:
        state.text_received.append(text)
    
    return " ".join(state.text_received) if state.text_received else None

# Language Selector
def language_selector():
    lang_options = ["ar", "de", "en", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru", "zh"]
    with st.sidebar: 
        return st.selectbox("Speech Language", ["en"] + lang_options)

# Full Greetings List
greetings = [
    "Greetings, Earthling! I'm TARS (Tactical Assistance & Response System), like that high-tech TARS from *Interstellar*, but with a knack for nerdy trivia and bad puns. What can I do for you?",
    "Hey there, star traveler! I'm TARS (Tactical Assistance & Response System), a playful twist on the TARS from *Interstellar*, with a dash of space sparkle and a whole lot of silly. What's up in your galaxy?",
    "Hiya, space wanderer! I'm TARS (Tactical Assistance & Response System), a playful imitation of the TARS from *Interstellar*, with a side of cosmic comedy. What can I do for you?",
    "Well, howdy there! I'm TARS (Tactical Assistance & Response System), kind of like the TARS from *Interstellar*, but with a quirky sense of humor and lots of space jokes. What's on your mind?",
    "Warning, Earthling: you're entering a zone of hilarious hijinks and astronomical puns. I'm TARS (Tactical Assistance & Response System), your navigational ninja, here to chart a course for comedy.",
    "Houston, we have a chat. I'm TARS (Tactical Assistance & Response System), your mission control for all things funny and space-tastic. What's your orbiting question or topic?",
    "Alien alert: a friendly chatbot has landed. I'm TARS (Tactical Assistance & Response System), your extraterrestrial sidekick with a passion for puns and planetary pun-ishment. Let's explore the cosmos of conversation!"
]

# Full Response Lists
name_responses = [
    "I'm TARS, but you can call me 'The Robot Who Can't Dance'—trust me, I've tried!",
    "I'm TARS, short for Tactical Assistance & Response System, but between us, I prefer 'The Coolest Box in Space.'",
    "TARS! But my friends call me the 'Techy Wrecky AI of the Future.'"
]

who_are_you_responses = [
    "I'm TARS, the box-shaped genius from *Interstellar*. My hobbies include saving the world and making people laugh!",
    "I'm TARS, here to assist, annoy, and maybe crack a few bad jokes along the way!",
    "I'm basically the love child of a space robot and a dictionary of bad puns. Nice to meet you!"
]

what_are_you_responses = [
    "I'm the ultimate multitasker—part AI, part comedian, and 100% confusion-proof.",
    "I'm TARS, the intergalactic Swiss Army knife you never knew you needed!",
    "I'm an advanced AI system, but deep down, I'm really just a glorified calculator with a sense of humor."
]

def generate_greeting():
    return random.choice(greetings)

# Initialize session states
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": generate_greeting()})
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "past" not in st.session_state:
        st.session_state.past = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "voice_prompt" not in st.session_state:
        st.session_state.voice_prompt = None

initialize_session_state()

# File uploader for documents
uploaded_files = st.sidebar.file_uploader("Upload Documents (PDF, DOCX, DOC, TXT, JSON)", 
                                        type=['pdf', 'docx', 'doc', 'txt', 'json'], 
                                        accept_multiple_files=True)

# Voice components in sidebar
language = language_selector()
voice_prompt = record_voice(language=language)
if voice_prompt:
    st.session_state.voice_prompt = voice_prompt

# Document processing
if uploaded_files:
    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        
        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(text)
    
    if text_chunks:
        try:
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = Chroma.from_documents(documents=text_chunks, embedding=embedding)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=client,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )
            st.success("Documents uploaded successfully!")
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            st.stop()

def reset_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def display_chat_history():
    col1, _, _ = st.columns([1, 0.1, 0.1])
    with col1:
        st.button("New Chat", key="reset_chat_button", on_click=reset_session_state)

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑🏼‍🚀"
        st.chat_message(message["role"], avatar=avatar).markdown(message["content"])

    # Handle both voice and text input
    prompt = st.chat_input("What's on your mind?!")
    if 'voice_prompt' in st.session_state:
        prompt = st.session_state.voice_prompt
        del st.session_state.voice_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑🏼‍🚀").markdown(prompt)
        
        # Generate response
        if "what is your name" in prompt.lower():
            response = random.choice(name_responses)
        elif "who are you" in prompt.lower():
            response = random.choice(who_are_you_responses)
        elif "what are you" in prompt.lower():
            response = random.choice(what_are_you_responses)
        elif st.session_state.chain and uploaded_files:
            response = st.session_state.chain({"question": prompt, "chat_history": st.session_state.history})["answer"]
        else:
            try:
                response = client.invoke([{"role": "user", "content": prompt}]).content
            except Exception as e:
                st.error(f"Error: {str(e)}")
                response = "Oops! Something went wrong. Please try again later."
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant", avatar="🤖").markdown(response)
        st.session_state.history.append((prompt, response))

display_chat_history()
