from llama_cpp import Llama

class LlamaChat:
    def __init__(self, model_path, n_gpu_layers=24, n_threads=12, n_ctx=4096):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx,
            verbose=False
        )

    def generate(self, prompt, max_tokens=500, temperature=0.5, top_p=0.9, repeat_penalty=1.1):
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty, 
            stop=["<|user|>", "<|assistant|>", "<|system|>"]
        )
        return output["choices"][0]["text"].strip()


