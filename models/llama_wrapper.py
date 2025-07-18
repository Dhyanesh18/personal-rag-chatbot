from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler

class LlamaChat:
    def __init__(self, model_path, n_gpu_layers=28, n_threads=12, n_ctx=8192): 
        handler = Llama3VisionAlphaChatHandler(
            clip_model_path="./models/llama-3-vision-alpha-mmproj-f16.gguf"
        )
        self.llm = Llama(
            model_path=model_path,
            chat_handler=handler,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx
        )

    def generate(self, prompt, max_tokens=512, temperature=0.4, top_p=0.85,  repeat_penalty=1.2, stop_tokens=["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]):
        stream = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop_tokens,
            stream=True  # Streaming mode
        )

        # Collect streamed output
        output_text = ""
        for part in stream:
            output_text += part["choices"][0]["text"]
            print(part["choices"][0]["text"], end="", flush=True)  # optional: print live

        return output_text.strip()
    
    def tokenize(self, text: str) -> list:
        return self.llm.tokenize(text.encode("utf-8"))