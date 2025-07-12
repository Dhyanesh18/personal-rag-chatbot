def generate_session_summary(messages: list, llama_instance) -> str:
    """Use LLM to generate intelligent session summary"""
    if not messages:
        return "Empty session"
    conversation = ""
    for user_query, assistant_response in messages:
        conversation += f"User: {user_query}\nJARVIS: {assistant_response}\n\n"
    summary_prompt = f"""
You are JARVIS, an AI assistant. You are tasked with writing a concise, factual summary of the following conversation.

Do not include greetings, sign-offs, disclaimers, or editorial comments like "Please feel free to edit".

ONLY return the summary. Be neutral and strictly factual. No imaginary context. No speculative language.

Conversation:
{conversation}

Summary (start below this line):
"""
    
    num_messages = len(messages)
    if num_messages <= 3:
        summary_max_tokens = 192
    elif num_messages <= 10:
        summary_max_tokens = 384
    else:
        summary_max_tokens = 512

    response = llama_instance.generate(summary_prompt, max_tokens=summary_max_tokens, temperature=0.3, top_p=0.85, repeat_penalty=1.2, stop_tokens=["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>","Note:", "**End of Summary**","\n\n\n"])

    return response