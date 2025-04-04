import os
import sys
import tempfile
import random
import asyncio
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import dotenv
from streamlit_mic_recorder import speech_to_text

# SQLite configuration fix
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

# Environment setup
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
dotenv.load_dotenv()

# Groq client setup
groq_api_key = os.getenv("GROQ_API_KEY")
client = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,
    groq_api_key=groq_api_key,
    request_timeout=30
)

# UI Configuration
st.set_page_config(page_icon="🌌", layout="wide", page_title="TARS")

def display_image():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("TARS.png", width=220)

# Original Personality Responses
greetings = [
    "Greetings, Earthling! I'm TARS (Tactical Assistance & Response System) ready to assist!",
    "Interstellar navigator TARS reporting for duty!",
    "Warning: Humor modules activated. TARS ready to assist!"
]

name_responses = [
    "I'm TARS - Think Awesome Robot Sidekick!",
    "TARS: Tactical Assistance & Response System!",
    "I go by TARS - your personal AI companion!"
]

# Session State Management
def initialize_session_state():
    session_defaults = {
        "history": [],
        "messages": [{"role": "assistant", "content": random.choice(greetings)}],
        "chain": None,
        "voice_prompt": None,
        "text_prompt": None,
        "uploaded_files": None,
        "recording": False,
        "processing": False
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

initialize_session_state()

# Document Processing
def process_documents(uploaded_files):
    try:
        documents = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                loader = None
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file.name)
                elif file.name.endswith((".docx", ".doc")):
                    loader = Docx2txtLoader(temp_file.name)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(temp_file.name)
                if loader:
                    documents.extend(loader.load())
                os.remove(temp_file.name)
        
        if documents:
            text_splitter = CharacterTextSplitter(
                chunk_size=1024, chunk_overlap=256
            )
            splits = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(splits, embeddings)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                client,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                memory=ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            )
            st.success("Documents loaded successfully!")
    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        st.stop()

# Voice Input Handling (Fixed Syntax)
async def handle_voice_input():
    language_map = {
        "English": "en", "Arabic": "ar", "German": "de", "Spanish": "es",
        "French": "fr", "Italian": "it", "Japanese": "ja", "Dutch": "nl",
        "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Chinese": "zh"
    }
    
    with st.sidebar:
        st.subheader("Voice Controls")
        lang = st.selectbox("Speech Language", list(language_map.keys()))
        
        col1, col2 = st.columns(2)
        with col1:
            start_rec = st.button("🎤 Start", disabled=st.session_state.processing)
        with col2:
            stop_rec = st.button("⏹️ Stop", disabled=not st.session_state.recording)
        
        if start_rec:
            st.session_state.recording = True
        if stop_rec:
            st.session_state.recording = False
        
        if st.session_state.recording:
            try:
                voice_prompt = speech_to_text(  # Fixed parenthesis
                    language=language_map[lang],
                    start_prompt="",
                    stop_prompt="",
                    use_container_width=True,
                    just_once=False,
                    key="voice_input",
                    callback=lambda: st.session_state.update(recording=False)
                
                if voice_prompt and voice_prompt.strip():
                    st.session_state.voice_prompt = voice_prompt.strip()
                    st.session_state.processing = True
                    await asyncio.sleep(0.5)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Voice recognition error: {str(e)}")
                st.session_state.recording = False

# Main Processing
def process_input(prompt):
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner('TARS is analyzing...'):
            lower_prompt = prompt.lower()
            if any(q in lower_prompt for q in ["your name", "who are you"]):
                response = random.choice(name_responses)
            elif st.session_state.chain and st.session_state.uploaded_files:
                result = st.session_state.chain({
                    "question": prompt,
                    "chat_history": st.session_state.history
                })
                response = result["answer"]
            else:
                response = client.invoke([HumanMessage(content=prompt)]).content
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.session_state.history.append((prompt, response))
    except Exception as e:
        st.error(f"Processing error: {str(e)}")

# Main Application
async def main():
    display_image()
    
    with st.sidebar:
        st.title("Configuration")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True
        )
        st.session_state.uploaded_files = uploaded_files
        
        if uploaded_files:
            process_documents(uploaded_files)
        
        if st.button("🧹 New Chat"):
            st.session_state.clear()
            initialize_session_state()
            st.rerun()
    
    await handle_voice_input()
    
    if text_prompt := st.chat_input("Type your message..."):
        st.session_state.text_prompt = text_prompt.strip()
    
    # Process inputs
    if st.session_state.get('voice_prompt'):
        prompt = st.session_state.pop('voice_prompt')
        process_input(prompt)
        st.session_state.processing = False
        st.rerun()
    
    if 'text_prompt' in st.session_state:
        prompt = st.session_state.pop('text_prompt')
        process_input(prompt)
        st.rerun()
    
    # Display messages
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👨🚀"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

if __name__ == "__main__":
    asyncio.run(main())
