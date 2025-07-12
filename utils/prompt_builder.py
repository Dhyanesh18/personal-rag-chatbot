import time

def build_prompt(system_prompt, relevant_summaries, current_session_context, user_input):
    """
    Build a prompt for Llama 3.1 with:
    - System prompt with timestamp
    - Relevant session summaries from memory
    - Current session context
    - User input
    """
    timestamp = time.strftime("%A, %B %d, %Y at %I:%M %p", time.localtime())
    
    # Build enhanced system prompt with memory context
    enhanced_system_prompt = f"{system_prompt}\nCurrent local time is {timestamp}."
    
    # Add relevant memories if available
    if relevant_summaries:
        enhanced_system_prompt += "\n\nRelevant memories from past sessions:"
        for summary in relevant_summaries:
            relevance_score = summary.get('relevance_score', 0)
            if relevance_score > 0.3:  # Only include reasonably relevant summaries
                enhanced_system_prompt += f"\n- {summary['content']}"
    
    # Start with the system prompt
    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{enhanced_system_prompt}<|eot_id|>"
    )
    
    # Add current session context if available
    if current_session_context:
        # Parse the session context and convert to Llama format
        prompt += _parse_session_context_to_llama_format(current_session_context)
    
    # Add final user input and prepare for assistant response
    prompt += (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return prompt

def _parse_session_context_to_llama_format(session_context):
    """
    Parse session context string and convert to Llama chat format.
    
    Expected format from SessionManager:
    Previous conversation:
    
    User: message1
    Jarvis: response1
    
    User: message2
    Jarvis: response2
    """
    if not session_context.strip():
        return ""
    
    llama_format = ""
    lines = session_context.strip().split('\n')
    
    current_user_msg = ""
    current_assistant_msg = ""
    
    for line in lines:
        line = line.strip()
        
        # Skip headers and empty lines
        if not line or line.startswith("Previous conversation:") or line.startswith("Recent conversation:") or line.startswith("[Earlier in this conversation"):
            continue
        
        if line.startswith("User: "):
            # If we have a previous complete exchange, add it
            if current_user_msg and current_assistant_msg:
                llama_format += (
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{current_user_msg}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{current_assistant_msg}<|eot_id|>"
                )
                current_assistant_msg = ""
            
            current_user_msg = line[6:]  # Remove "User: " prefix
            
        elif line.startswith("Jarvis: "):
            if current_user_msg:  # Only if we have a user message
                current_assistant_msg = line[8:]  # Remove "Jarvis: " prefix
        
        elif current_user_msg and not line.startswith("User: ") and not line.startswith("Jarvis: "):
            # Continue previous message (multi-line)
            if current_assistant_msg:
                current_assistant_msg += " " + line
            else:
                current_user_msg += " " + line
    
    # Add the last exchange if complete
    if current_user_msg and current_assistant_msg:
        llama_format += (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{current_user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{current_assistant_msg}<|eot_id|>"
        )
    
    return llama_format