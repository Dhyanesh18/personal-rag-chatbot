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

# IMPORTANT: llama-cpp-python must be installed with CUDA support for SM86
pip uninstall llama-cpp-python
set CMAKE_ARGS="-DGGML_CUDA=on;-DCMAKE_CUDA_ARCHITECTURES=86"
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir --force-reinstall

# Prepare models
mkdir -p models
# Download and place:
# - llama-3.1-8b-instruct-q4_k_m.gguf
# - llama-3-vision-alpha-mmproj-f16.gguf
```

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

```bash
python main.py
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
