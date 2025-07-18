"""
jarvis_assistant.py

JARVIS: An AI Assistant modeled after Tony Stark's digital assistant.
Combines:
    - Open-source LLaMA model (through LlamaChat wrapper)
    - Hybrid Retrieval (BM25 + Dense Embeddings + RRF)
    - Persistent memory (ChromaDB + Elasticsearch)
    - Session management and logging

Behavior:
    - Always refers to itself as 'JARVIS'
    - Always refers to the user as 'Sir'
    - Never hallucinates or invents ungrounded context
"""

from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from memory.logger import JSONLogger
from memory.session_manager import SessionManager
from utils.prompt_builder import build_prompt
from utils.time_utils import get_time_based_greeting
from utils.estimate_tokens import estimate_tokens
from utils.generate_summary import generate_session_summary

from elasticsearch import Elasticsearch
from hybrid_pipeline import HybridRetrievalPipeline


# ======================
# SYSTEM PROMPT
# ======================

SYSTEM_PROMPT = """
You are JARVIS, an AI assistant modeled after Tony Stark's advanced digital assistant.

Your role is to assist the user respectfully, efficiently, and professionally.

Behavior rules:
- Always refer to yourself as 'JARVIS'
- Always refer to the user as 'Sir'
- Never assume or invent any personal details or context
- Never make up facts, events, or relationships
- If you lack information, respond: "I don't have that information in my memory banks, Sir."

You may use:
- Memories from previous sessions
- Current session messages

Your response must be:
- Concise and grounded only in the provided context
- Respectful and professional
- Truthful and non-speculative
"""


# ======================
# INITIALIZATION
# ======================

llama = LlamaChat(model_path="./models/llama-3.1-8b-instruct-q4_k_m.gguf")
embedder = Embedder()
memory = MemoryStore()
logger = JSONLogger()
session_manager = SessionManager()
session_id = session_manager.get_or_create_session()

es_client = Elasticsearch("http://localhost:9200")
pipeline = HybridRetrievalPipeline(es_client=es_client, top_k=50)

print("Jarvis AI Assistant - Initializing...")
greeting = f"{get_time_based_greeting()}, Sir. JARVIS is online and ready to assist you."
print(f"JARVIS: {greeting}\n")
logger.log("Session started", greeting)


# ======================
# MAIN INTERACTION LOOP
# ======================

try:
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue

        # ---- SPECIAL COMMANDS ----
        if user_input.lower() == "/reset":
            memory.reset()
            print("Memory banks cleared, Sir!")
            continue

        if user_input.lower() == "/stats":
            stats = memory.get_stats()
            session_stats = session_manager.get_session_stats()
            print(f"Memory Bank: {stats['total_summaries']} session summaries stored")
            print(f"Current Session: {session_stats.get('message_count', 0)} messages, {session_stats.get('total_tokens', 0)} tokens")
            continue

        if user_input.lower() == "/cleanup":
            memory.cleanup_old_summaries(keep_recent=30)
            print("Memory optimization complete, Sir!")
            continue

        if user_input.lower() == "/rebuild-bm25":
            pipeline.rebuild_bm25_index()
            print("BM25 index rebuilt, Sir!")
            continue

        # ---- SESSION CHECK ----
        if not session_manager._is_session_active() or session_manager.should_end_session(user_input):
            messages = session_manager.get_all_messages()
            if messages:
                print("Generating session summary, Sir...")
                session_summary = generate_session_summary(messages, llama)

                session_data = session_manager.end_session(session_summary)

                # Store in ChromaDB (dense)
                summary_embedding = embedder.get_embedding(session_summary)
                memory.store(session_summary, summary_embedding, session_data)

                # Store in Elasticsearch (BM25)
                es_client.index(
                    index="summaries",
                    document={"text": session_summary, "metadata": session_data}
                )

                print("Session summary stored in ChromaDB + Elasticsearch, Sir!")
            else:
                session_manager.end_session()
                print("Session ended — no summary needed for empty session, Sir.")

            # Start new session
            session_id = session_manager.get_or_create_session()
            continue

        # ---- HYBRID RETRIEVAL ----
        relevant_summaries = pipeline.retrieve(user_input, final_k=5)

        # ---- CURRENT SESSION CONTEXT ----
        current_session_context = session_manager.get_session_context()

        # ---- BUILD PROMPT ----
        prompt = build_prompt(
            SYSTEM_PROMPT,
            relevant_summaries,
            current_session_context,
            user_input
        )

        # ---- TOKEN ESTIMATION ----
        estimated_tokens = estimate_tokens(prompt, llama)

        # ---- GENERATE RESPONSE ----
        response = llama.generate(prompt)
        print(f"JARVIS: {response}\n")

        # ---- LOG & SAVE ----
        logger.log(user_input, response)
        session_manager.add_message(user_input, response, estimated_tokens)

except KeyboardInterrupt:
    print("\nInterrupt received, Sir. Attempting emergency session save...")

    try:
        messages = session_manager.get_all_messages()
        if messages:
            session_summary = generate_session_summary(messages, llama)
            session_data = session_manager.end_session(session_summary)
            summary_embedding = embedder.get_embedding(session_summary)
            memory.store(session_summary, summary_embedding, session_data)
            es_client.index(
                index="summaries",
                document={"text": session_summary, "metadata": session_data}
            )
            print("Emergency session save complete, Sir!")
    except Exception as e:
        print(f"Could not save session data, Sir. Reason: {e}")

    logger.log_session_end()

finally:
    if 'llama' in locals():
        del llama
    print("Resources cleaned up, Sir. Goodbye.")
