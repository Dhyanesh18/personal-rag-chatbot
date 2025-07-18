#  JARVIS - Multimodal RAG Chatbot with Session Memory

This repository implements **JARVIS**, a llama‑cpp‑python-powered AI assistant featuring:

* **Hybrid Retrieval-Augmented Generation (RAG)**

  * In-memory **ChromaDB** for vector-based semantic session summary retrieval
  * **SQLite** for structured session logs and metadata
  * Fused past-memory + live session context to build each prompt
  * **all-MiniLM-L6-v2** for dense retrieval and **Elasticsearch** for sparse retrieval, combined using RRF 

* **Multimodal support**

  * Uses `llama-3.1-8b-instruct-q4_k_m.gguf` for text
  * Integrates `llama-3-vision-alpha-mmproj-f16.gguf` for image understanding
  * The `llama-3-vision-alpha-mmproj-f16.gguf` contains a SigLIP Vision model and an adapter that projects the features into LLaMa's internal representation
  * Allows both text-only and image+text prompts

* **Intelligent sessions**

  * Auto-detects session end or reset commands
  * Generates and persists session summaries via RAG
  * Limits context weight based on token budgets and relevance

* **Resource-aware runtime**

  * GPU utilization on RTX 3050 Ti 4 GB via cuda-compiled llama–cpp–python
  * 16 GB system RAM
  * Efficient quantized model (Q4\_K\_M)

* **Interactive Operational Flow**

  * Python CLI loop for user interaction
  * Custom commands: `/reset`, `/stats`, `/cleanup`
  * All within `while True`—no CLI flags

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/your-repo/jarvis-rag-chatbot.git
cd jarvis-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Prepare models
mkdir -p models
# Download and place:
# - llama-3.1-8b-instruct-q4_k_m.gguf
# - llama-3-vision-alpha-mmproj-f16.gguf
```

### 🔧 Build `llama-cpp-python` with CUDA 12.6

This project requires a custom build of `abetlen/llama-cpp-python` with CUDA support.

#### 1. Prerequisites
- CUDA Toolkit 12.6 (nvcc, etc.) installed and in PATH
- C/C++ compiler + CMake installed

#### 2. Clone the repo
```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
```
#### 3. Build and install

Linux / Mac:
```
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CMAKE=1
export CMAKE_ARGS="-DGGML_CUDA=on"
pip install . --upgrade --force-reinstall --no-cache-dir
```
Windows (CMD / PowerShell):

```
set CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set PATH=%CUDA_HOME%\bin;%PATH%
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=on
pip install . --upgrade --force-reinstall --no-cache-dir
```
#### 4. Test your installation
```
from llama_cpp import Llama  
llm = Llama(model_path="model.gguf", n_gpu_layers=-1, verbose=True)  
```
Look for gpu initialization logs

---

## Files & Directory Structure

```
.
├── main.py                 # Entry point, CLI loop
├── models/                # GGUF model files
│   ├── llama-3.1-8b-instruct-q4_k_m.gguf
│   └── llama-3-vision-alpha-mmproj-f16.gguf
├── models/llama_wrapper.py # LlamaChat class with multimodal support
├── memory/                 # RAG and memory modules
│   ├── embedder.py         # OpenAI embedding wrapper or local LLM
│   ├── memory_store.py     # ChromaDB handling  
│   ├── session_manager.py  # Context/session logic
│   └── logger.py           # JSON logging
├── utils/                  # Assistive functions
│   ├── prompt_builder.py
│   ├── estimate_tokens.py
│   ├── generate_summary.py
│   └── time_utils.py
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Usage
```
docker-compose up -d
```
```bash
python main.py
```

To Exit the application, use Ctrl + C. Then stop the elasticsearch containers

```
docker-compose down
```

### Interactive Commands

* `/reset` — Clear memory and start fresh
* `/stats` — View memory & session stats
* `/cleanup` — Prune old summaries (keep recent 30)

Sessions auto-end on inactivity or defined triggers (`SessionManager.should_end_session()`), prompting summary generation and storage.

---

## Customization

* Adjust GPU layer offload: `n_gpu_layers`, `n_threads`, `n_ctx` in `LlamaChat.__init__`
* Change segmentation behavior in `session_manager`
* Swap embedder for different semantic backends (OpenAI or embedding LLM)

---

## Troubleshooting

* If you see CUDA kernel errors, rebuild llama-cpp-python per the installation steps above
* For memory DB issues, inspect `memory_store.py` / `SessionManager` logs

---

## License

\[MIT License] — see [LICENSE](LICENSE)

---
