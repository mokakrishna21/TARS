import os
import sys
import tempfile
import random
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
import dotenv
from streamlit_mic_recorder import speech_to_text

# Environment setup
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
dotenv.load_dotenv()

# Groq client setup
groq_api_key = os.getenv("GROQ_API_KEY")
client = ChatGroq(
    model="llama-3.3-70b",
    temperature=0.5,
    groq_api_key=groq_api_key
)

# UI Configuration
st.set_page_config(page_icon="🌌", layout="wide", page_title="TARS")

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
        start_prompt="🎤 Speak",
        stop_prompt="⏹️ Stop",
        language=language,
        use_container_width=True,
        just_once=True,
        key=f"recorder_{language}"
    )
    if text and text.strip():
        state.text_received.append(text.strip())
    return " ".join(state.text_received) if state.text_received else ""

# Language Selector
def language_selector():
    language_map = {
        "English": "en", "Arabic": "ar", "German": "de", "Spanish": "es",
        "French": "fr", "Italian": "it", "Japanese": "ja", "Dutch": "nl",
        "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Chinese": "zh"
    }
    return st.selectbox("Speech Language", options=list(language_map.keys()))

# Your Original Greetings & Responses (Preserved Exactly)
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
        "last_voice": None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

initialize_session_state()

# Sidebar Components
with st.sidebar:
    uploaded_files = st.file_uploader("Upload Documents", 
                                    type=['pdf', 'docx', 'doc', 'txt'], 
                                    accept_multiple_files=True)
    
    lang_name = language_selector()
    language_map = {
        "English": "en", "Arabic": "ar", "German": "de", "Spanish": "es",
        "French": "fr", "Italian": "it", "Japanese": "ja", "Dutch": "nl",
        "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Chinese": "zh"
    }
    current_voice = record_voice(language=language_map[lang_name])
    if current_voice.strip():
        st.session_state.last_voice = current_voice.strip()

# Document Processing
if uploaded_files:
    text = []
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            loader = None
            if file_ext == ".pdf": loader = PyPDFLoader(temp_file.name)
            elif file_ext in [".docx", ".doc"]: loader = Docx2txtLoader(temp_file.name)
            elif file_ext == ".txt": loader = TextLoader(temp_file.name)
            if loader: 
                text.extend(loader.load())
            os.remove(temp_file.name)
    
    if text:
        text_splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=1024, chunk_overlap=256
        )
        text_chunks = text_splitter.split_documents(text)
        try:
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = Chroma.from_documents(text_chunks, embedding)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=client,
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )
        except Exception as e:
            st.error(f"Document error: {e}")
            st.stop()

# Input Handling
def handle_inputs():
    # Process text input
    text_prompt = st.chat_input("Type your message...")
    if text_prompt and text_prompt.strip():
        st.session_state.messages.append({"role": "user", "content": text_prompt.strip()})
        generate_response(text_prompt.strip())
    
    # Process voice input separately
    if 'last_voice' in st.session_state:
        voice_prompt = st.session_state.pop('last_voice')
        if voice_prompt and voice_prompt.strip():
            st.session_state.messages.append({"role": "user", "content": voice_prompt})
            generate_response(voice_prompt)

def generate_response(prompt):
    if not prompt:
        return
    
    with st.spinner('TARS is thinking...'):
        try:
            lower_prompt = prompt.lower()
            if "what is your name" in lower_prompt:
                response = random.choice(name_responses)
            elif "who are you" in lower_prompt:
                response = random.choice(who_are_you_responses)
            elif "what are you" in lower_prompt:
                response = random.choice(what_are_you_responses)
            elif st.session_state.chain and uploaded_files:
                response = st.session_state.chain({"question": prompt, "chat_history": st.session_state.history})["answer"]
            else:
                response = client.invoke([{"role": "user", "content": prompt}]).content
        except Exception as e:
            response = f"Oops! Something went wrong: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.history.append((prompt, response))

# Display Chat
def display_chat():
    st.button("New Chat", on_click=lambda: st.session_state.clear())
    
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑🏼‍🚀"
        st.chat_message(message["role"], avatar=avatar).markdown(message["content"])
    
    handle_inputs()

display_chat()
