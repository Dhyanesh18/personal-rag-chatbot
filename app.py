import streamlit as st
import json
from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from memory.logger import JSONLogger
from utils.prompt_builder import build_prompt
from utils.time_utils import get_time_based_greeting
from utils.chat_log_utils import load_sessions, get_session_summaries

st.set_page_config(page_title="Jarvis", layout="wide")

@st.cache_resource
def load_llama():
    return LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")

@st.cache_resource
def load_embedder():
    return Embedder()

@st.cache_resource
def load_memory():
    return MemoryStore()

@st.cache_resource
def load_logger():
    return JSONLogger()

llama = load_llama()
embedder = load_embedder()
memory = load_memory()
logger = load_logger()


system_prompt = "You are Jarvis, a helpful, intelligent AI assistant. Always refer to yourself as 'Jarvis' when speaking with the user. The user prefers to be addressed only as 'Sir' — never use their real name, even if provided. Avoid mentioning the user's real name. Be concise, respectful, and professional in all responses. Provide accurate information based on the user's queries. If you don't know the answer, say 'I don't know, Sir'. Do not make up information."

# Load Sessions
sessions = load_sessions()
session_summaries = get_session_summaries(sessions)

# APP layout
st.title("Jarvis - Your Personal Assistant")

# Sidebar for chat history
st.sidebar.header("Chat History")
session_index = st.sidebar.selectbox("Select a session", range(len(session_summaries)), format_func=lambda i: session_summaries[i])

# Start New Chat
if st.sidebar.button("New Chat"):
    # Log session end of previous session if any chat exists
    if "chat_history" in st.session_state and len(st.session_state.chat_history) > 0:
        logger.log_session_end()
    
    # Clear chat history and start new session
    st.session_state.chat_history = []
    logger.log_session_start()
    greeting = f"{get_time_based_greeting()}, Sir. I am Jarvis. How may I assist you today?"
    st.session_state.chat_history.append({"role": "assistant", "content": greeting})
    logger.log("Session started", greeting)


# Display selected session chat history
if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
    greeting = f"{get_time_based_greeting()}, Sir. I am Jarvis. How may I assist you today?"
    st.session_state.chat_history = [{"role": "assistant", "content": greeting}]
    logger.log("Session started", greeting)
    if sessions:
        st.session_state.chat_history = [
            {"role": "user", "content": msg["user"]} if msg["type"] == "message" else
            {"role": "assistant", "content": msg["assistant"]} for msg in sessions[session_index] if msg["type"] == "message"
        ]

# Chat display
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").write(chat["content"])
        else:
            st.chat_message("assistant").write(chat["content"])

# User Input Box
user_input = st.chat_input("Type your message...")
if user_input:
    # Immediately show the user message
    st.session_state.chat_history.append({"role":"user", "content": user_input})
    user_msg = st.chat_message("user")
    user_msg.write(user_input)

    # Prepare placeholder for assistant reply
    assistant_msg = st.chat_message("assistant")

    # Run embedding and retrieval before generating
    query_embed = embedder.get_embedding(user_input)
    relevant = memory.retrieve(query_embed)
    prompt = build_prompt(system_prompt, relevant, user_input)

    # Show spinner while generating
    with st.spinner("Jarvis is thinking..."):
        response = llama.generate(prompt)

    # Append assistant response to chat history and display it
    st.session_state.chat_history.append({"role":"assistant", "content": response})
    assistant_msg.write(response)

    # Log and store as usual
    logger.log(user_input, response)
    memory.store(user_input, response, query_embed)