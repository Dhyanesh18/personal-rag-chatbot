def estimate_tokens(text: str, llama_instance) -> int:
    """Accurate token count using LLaMA's tokenizer"""
    return len(llama_instance.tokenize(text))