import os
import sys
import tempfile
import pandas as pd
import streamlit as st
from io import StringIO
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
from fbprophet import Prophet
import matplotlib.pyplot as plt
import dotenv

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
st.set_page_config(page_icon="🌌", layout="wide", page_title="TARS with RAG")

# Display Image
def display_image(image_path: str):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        st.image(image_path, width=220)

display_image("TARS.png")

# Initialize session states
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello there! I'm TARS (Tactical Assistance & Response System), a bootleg version of the TARS from Interstellar. How can I help you?"})
    if "generated" not in st.session_state:
        st.session_state.generated = ["Hello! Feel free to ask me any questions."]
    if "past" not in st.session_state:
        st.session_state.past = ["Hey! 👋"]
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

initialize_session_state()

# File uploader for documents and CSV
uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True, key="file_uploader")

if uploaded_files:
    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        elif file_extension == ".csv":
            dataframe = pd.read_csv(temp_file_path)
            st.session_state.dataframe = dataframe
            st.write(dataframe)
            os.remove(temp_file_path)
            continue
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue
        
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
    
    if not text_chunks:
        st.error("No valid text chunks found. Please check your documents.")
        st.stop()
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # Use in-memory Chroma database instead of persistent storage
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

# Function to handle forecasting with Prophet
def forecast_with_prophet(df, periods=30):
    if 'Date' not in df.columns or 'Value' not in df.columns:
        return "Data must contain 'Date' and 'Value' columns."
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.rename(columns={'Date': 'ds', 'Value': 'y'})
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    fig = model.plot(forecast)
    st.pyplot(fig)
    
    return forecast

# Function to handle CSV queries
def handle_csv_query(query, dataframe):
    query = query.lower()
    
    if "forecast" in query:
        periods = int(query.split("forecast")[1].strip().split()[0])
        forecast = forecast_with_prophet(dataframe, periods=periods)
        if isinstance(forecast, str):
            return forecast
        else:
            return "Forecasting complete. Check the plot."
    
    if "summary" in query:
        return dataframe.describe().to_string()

    if "head" in query:
        return dataframe.head().to_string()

    if "columns" in query:
        return ', '.join(dataframe.columns)

    if "info" in query:
        buffer = StringIO()
        dataframe.info(buf=buffer)
        return buffer.getvalue()

    if "plot" in query:
        columns = query.split("plot")[1].strip().split()
        if len(columns) != 2:
            return "Please specify exactly two columns to plot."
        if all(col in dataframe.columns for col in columns):
            fig, ax = plt.subplots()
            dataframe.plot(x=columns[0], y=columns[1], kind='scatter', ax=ax)
            st.pyplot(fig)
            return "Plot displayed."
        else:
            return "Specified columns not found in the dataframe."

    return "I can only provide summary statistics, basic plots, or column-wise analysis."

# Function to handle built-in queries about the bot
def handle_builtin_query(query):
    query = query.lower()
    if "what is your name" in query or "who are you" in query or "what are you" in query:
        return "Hello there! I'm TARS (Tactical Assistance & Response System), a bootleg version of the TARS from Interstellar. How can I help you?"
    return None

# Function to reset the session state
def reset_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

# Display chat history and handle inputs
def display_chat_history():
    col1, _, _ = st.columns([1, 0.1, 0.1])
    with col1:
        st.button("New Chat", key="reset_chat_button_2", on_click=reset_session_state)

    for message in st.session_state.messages:
        if message["role"] != "system":
            avatar = "🤖" if message["role"] == "assistant" else "🧑🏼‍🚀"
            st.chat_message(message["role"], avatar=avatar).markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑🏼‍🚀").markdown(prompt)
        
        # Handle built-in queries
        response = handle_builtin_query(prompt)
        
        if response is None:
            # Retrieve and generate response
            if st.session_state.chain and uploaded_files:
                response = st.session_state.chain({"question": prompt, "chat_history": st.session_state.history})["answer"]
            elif st.session_state.dataframe is not None:
                response = handle_csv_query(prompt, st.session_state.dataframe)
            else:
                try:
                    response = client.invoke([{"role": "user", "content": prompt}]).content
                except Exception as e:
                    st.error(f"Error: {str(e)}", icon="🚨")
                    response = "Oops, something went wrong!"

        st.chat_message("assistant", avatar="🤖").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Show chat and process inputs
display_chat_history()
