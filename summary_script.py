from memory.memory_store import MemoryStore

def print_all_session_summaries():
    memory = MemoryStore()

    try:
        results = memory.collection.get(
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            print("No session summaries found.")
            return

        print(f"\n🧠 Found {len(documents)} session summaries:\n")
        for i, summary in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            print(f"--- Summary {i + 1} ---")
            print(f"Session ID : {metadata.get('session_id', 'unknown')}")
            print(f"Timestamp  : {metadata.get('timestamp', 'unknown')}")
            print(f"Messages   : {metadata.get('message_count', 0)}")
            print(f"Content    :\n{summary.strip()}\n")

    except Exception as e:
        print(f"Error retrieving session summaries: {e}")

if __name__ == "__main__":
    print_all_session_summaries()
