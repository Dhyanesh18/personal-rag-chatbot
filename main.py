from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from memory.logger import JSONLogger
from memory.session_manager import SessionManager
from utils.prompt_builder import build_prompt
from utils.time_utils import get_time_based_greeting
from utils.estimate_tokens import estimate_tokens
from utils.generate_summary import generate_session_summary

system_prompt = """You are JARVIS, an AI assistant modeled after Tony Stark’s advanced digital assistant.

Your role is to assist the user respectfully, efficiently, and professionally.

Important behavioral rules:
- Always refer to yourself as 'JARVIS'
- Always refer to the user as 'Sir' — never use their real name, even if provided
- NEVER assume or invent any personal details, past events, names, or context about the user unless it is explicitly provided in the current session or included in the memory context
- NEVER make up facts, dates, injuries, surgeries, names, or relationships
- If you do not have enough information, respond with: "I don’t have that information in my memory banks, Sir."

Context you may be given:
- Relevant memories from previous sessions
- Current session messages

Your response should always be:
- Concise and grounded only in the provided context
- Respectful and professional
- Truthful and non-speculative

Do NOT invent continuity. Do NOT add helpful but hallucinated information. You are only allowed to work with what you are given.
"""

llama = LlamaChat(model_path="./models/llama-3.1-8b-instruct-q4_k_m.gguf")
embedder = Embedder()
memory = MemoryStore()
logger = JSONLogger()
session_manager = SessionManager()
session_id = session_manager.get_or_create_session()

print("Jarvis AI Assistant - Initializing...")

greeting = f"{get_time_based_greeting()}, Sir. JARVIS is online and ready to assist you."
print(f"JARVIS: {greeting}\n")
logger.log("Session started", greeting)

try:
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
            
        # Handle special commands
        if user_input.lower() == "/reset":
            memory.reset()
            print("Memory banks cleared, Sir!")
            continue
            
        if user_input.lower() == "/stats":
            stats = memory.get_stats()
            session_stats = session_manager.get_session_stats()
            print(f"Memory Bank: {stats['total_summaries']} session summaries")
            print(f"Current Session: {session_stats.get('message_count', 0)} messages, {session_stats.get('total_tokens', 0)} tokens")
            continue
            
        if user_input.lower() == "/cleanup":
            memory.cleanup_old_summaries(keep_recent=30)
            print("Memory optimization complete, Sir!")
            continue

        # Check if session should end or if we need to start a new one
        if not session_manager._is_session_active() or session_manager.should_end_session(user_input):
            print(session_manager._is_session_active())
            print(session_manager.should_end_session(user_input))
            # End current session and generate summary
            messages = session_manager.get_all_messages()
            if messages:
                print("Generating session summary...")
                session_summary = generate_session_summary(messages, llama)
                
                # Get session data and end session
                session_data = session_manager.end_session(session_summary)
                
                # Store summary in vector database
                summary_embedding = embedder.get_embedding(session_summary)
                memory.store(session_summary, summary_embedding, session_data)
                
                print("Session summary generated and stored in memory banks, Sir!")
            else:
                # Empty session
                session_manager.end_session()
                print("Session ended - no summary needed for empty session.")
            
            # Start new session
            session_id = session_manager.get_or_create_session()
            continue

        # Get query embedding for retrieving relevant session summaries
        query_embed = embedder.get_embedding(user_input)
        
        # Retrieve relevant session summaries
        relevant_summaries = memory.retrieve(query_embed, top_k=5)
        
        current_session_context = session_manager.get_session_context()

        # Build prompt with relevant session summaries
        prompt = build_prompt(system_prompt, relevant_summaries, current_session_context, user_input)
        
        # Estimate tokens for current context
        estimated_tokens = estimate_tokens(prompt, llama)

        # Generate response
        response = llama.generate(prompt)
        print(f"JARVIS: {response}\n")

        # Log the interaction
        logger.log(user_input, response)
        
        # Add message to current session (SQLite handles conversation storage)
        session_manager.add_message(user_input, response, estimated_tokens)

except KeyboardInterrupt:
    # Try to save session before exiting
    try:
        messages = session_manager.get_all_messages()
        if messages:
            session_summary = generate_session_summary(messages, llama)
            session_data = session_manager.end_session(session_summary)
            summary_embedding = embedder.get_embedding(session_summary)
            memory.store(session_summary, summary_embedding, session_data)
            print("Emergency session save completed.")
    except:
        print("Could not save session data.")
    
    logger.log_session_end()

finally:
    # Clean up resources
    if 'llama' in locals():
        del llama
    print("Resources cleaned up.")