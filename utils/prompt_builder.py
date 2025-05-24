import time

def build_prompt(system_prompt, context_list, user_input):
    timestamp = time.strftime("%A, %B %d, %Y at %I:%M %p", time.localtime())
    full_system_prompt = f"{system_prompt}\nCurrent local time is {timestamp}."
    
    prompt = f"<|system|>\n{full_system_prompt}\n"
    for ctx in context_list:
        prompt += f"<|user|>\n{ctx['user']}\n<|assistant|>\n{ctx['assistant']}\n"
    prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return prompt