import time

def build_prompt(system_prompt, context_list, user_input):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prompt = f"<|system|>\n{system_prompt}\n<|timestamp|>\n{timestamp}\n"
    for ctx in context_list:
        prompt += f"<|user|>\n{ctx['user']}\n<|assistant|>\n{ctx['assistant']}\n"
    prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return prompt
