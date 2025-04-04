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

# Groq client configuration
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
    "Greetings, Earthling! I’m TARS (Tactical Assistance & Response System), like that high-tech TARS from *Interstellar*, but with a knack for nerdy trivia and bad puns. What can I do for you?",
    "Hey there, star traveler! I’m TARS (Tactical Assistance & Response System), a playful twist on the TARS from *Interstellar*, with a dash of space sparkle and a whole lot of silly. What’s up in your galaxy?",
    "Hiya, space wanderer! I’m TARS (Tactical Assistance & Response System), a playful imitation of the TARS from *Interstellar*, with a side of cosmic comedy. What can I do for you?",
    "Well, howdy there! I’m TARS (Tactical Assistance & Response System), kind of like the TARS from *Interstellar*, but with a quirky sense of humor and lots of space jokes. What’s on your mind?",
    "Warning, Earthling: you're entering a zone of hilarious hijinks and astronomical puns. I'm TARS (Tactical Assistance & Response System), your navigational ninja, here to chart a course for comedy.",
    "Houston, we have a chat. I'm TARS (Tactical Assistance & Response System), your mission control for all things funny and space-tastic. What's your orbiting question or topic?",
    "Alien alert: a friendly chatbot has landed. I'm TARS (Tactical Assistance & Response System), your extraterrestrial sidekick with a passion for puns and planetary pun-ishment. Let's explore the cosmos of conversation!"
]

name_responses = [
    "I'm TARS, but you can call me 'The Robot Who Can’t Dance'—trust me, I’ve tried!",
    "I’m TARS, short for Tactical Assistance & Response System, but between us, I prefer 'The Coolest Box in Space.'",
    "TARS! But my friends call me the 'Techy Wrecky AI of the Future.'"
]

who_are_you_responses = [
    "I’m TARS, the box-shaped genius from *Interstellar*. My hobbies include saving the world and making people laugh!",
    "I’m TARS, here to assist, annoy, and maybe crack a few bad jokes along the way!",
    "I'm basically the love child of a space robot and a dictionary of bad puns. Nice to meet you!"
]

what_are_you_responses = [
    "I’m the ultimate multitasker—part AI, part comedian, and 100% confusion-proof.",
    "I’m TARS, the intergalactic Swiss Army knife you never knew you needed!",
    "I’m an advanced AI system, but deep down, I’m really just a glorified calculator with a sense of humor."
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
    text = []
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
            
        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(temp_file_path)
            elif file_ext == ".txt":
                loader = TextLoader(temp_file_path)
            
            if loader:
                text.extend(loader.load())
        finally:
            os.remove(temp_file_path)
    
    if text:
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1024,
            chunk_overlap=256
        )
        text_chunks = text_splitter.split_documents(text)
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = Chroma.from_documents(text_chunks, embeddings)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=client,
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                memory=ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            )
            st.success("Documents ready for queries!")
        except Exception as e:
            st.error(f"Document processing error: {str(e)}")
            st.stop()

# Voice Input Handling
async def handle_voice_input():
    with st.sidebar:
        st.subheader("Voice Controls")
        lang_name = language_selector()
        language_map = {
            "English": "en", "Arabic": "ar", "German": "de", "Spanish": "es",
            "French": "fr", "Italian": "it", "Japanese": "ja", "Dutch": "nl",
            "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Chinese": "zh"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Start Recording", key="start_rec"):
                st.session_state.recording = True
        with col2:
            if st.button("⏹️ Stop Recording", key="stop_rec"):
                st.session_state.recording = False
        
        if st.session_state.recording:
            st.info("Listening... Speak now!")
            try:
                voice_prompt = speech_to_text(
                    language=language_map[lang_name],
                    start_prompt="",
                    stop_prompt="",
                    use_container_width=True,
                    just_once=False,
                    key="voice_input",
                    callback=lambda: st.session_state.update(recording=False)
                
                if voice_prompt and voice_prompt.strip():
                    st.session_state.voice_prompt = voice_prompt.strip()
                    await asyncio.sleep(0.3)  # Debounce delay
                    st.rerun()
            except Exception as e:
                st.error(f"Voice recognition error: {str(e)}")
                st.session_state.recording = False

# Text Input Handling
def handle_text_input():
    if text_prompt := st.chat_input("Type your message..."):
        st.session_state.text_prompt = text_prompt.strip()

# Response Processing
def process_input(prompt):
    if not prompt:
        return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner('TARS is thinking...'):
        try:
            lower_prompt = prompt.lower()
            if "your name" in lower_prompt:
                response = random.choice(name_responses)
            elif "who are you" in lower_prompt:
                response = random.choice(who_are_you_responses)
            elif "what are you" in lower_prompt:
                response = random.choice(what_are_you_responses)
            elif st.session_state.chain and st.session_state.uploaded_files:
                result = st.session_state.chain({"question": prompt, "chat_history": st.session_state.history})
                response = result["answer"]
            else:
                response = client.invoke([HumanMessage(content=prompt)]).content
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.history.append((prompt, response))
        except Exception as e:
            response = f"System error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": response})

# Main Interface
async def main():
    display_image()
    
    with st.sidebar:
        st.title("Configuration")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True
        )
        if uploaded_files:
            process_documents(uploaded_files)
            st.session_state.uploaded_files = uploaded_files
        
        if st.button("🧹 New Chat", key="new_chat"):
            st.session_state.clear()
            initialize_session_state()
            st.rerun()
    
    await handle_voice_input()
    handle_text_input()
    
    # Process inputs
    if st.session_state.get('voice_prompt'):
        prompt = st.session_state.pop('voice_prompt')
        process_input(prompt)
        st.rerun()
    
    if 'text_prompt' in st.session_state:
        prompt = st.session_state.pop('text_prompt')
        process_input(prompt)
        st.rerun()
    
    # Display messages
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑🏼‍🚀"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

if __name__ == "__main__":
    asyncio.run(main())
