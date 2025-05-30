from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from memory.logger import JSONLogger
from utils.prompt_builder import build_prompt
from utils.time_utils import get_time_based_greeting

system_prompt = "You are Jarvis, a helpful, intelligent AI assistant. Always refer to yourself as 'Jarvis' when speaking with the user. The user prefers to be addressed only as 'Sir' — never use their real name, even if provided. Avoid mentioning the user's real name. Be concise, respectful, and professional in all responses. Provide accurate information based on the user's queries. If you don't know the answer, say 'I don't know, Sir'. Do not make up information."

llama = LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
embedder = Embedder()
memory = MemoryStore()
logger = JSONLogger()

print("Chat with Mistral (type Ctrl+C to stop, or type /reset to clear memory)")

greeting = f"{get_time_based_greeting()}, Sir. I am Jarvis. How may I assist you today?"
print(f"Assistant: {greeting}\n")
logger.log("Session started", greeting)

try:
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "/reset":
            memory.reset()
            print("Memory cleared!")
            continue

        query_embed = embedder.get_embedding(user_input)
        relevant = memory.retrieve(query_embed)

        prompt = build_prompt(system_prompt, relevant, user_input)
        print(f"\nPrompt:\n{prompt}\n")

        response = llama.generate(prompt)

        print(f"Assistant: {response}\n")

        logger.log(user_input, response)

        memory.store(user_input, response, query_embed)

except KeyboardInterrupt:
    logger.log_session_end()
    print("\nExiting chat. Bye!")
