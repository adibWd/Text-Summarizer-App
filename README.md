# Text Summarizer App

A web-based text summarization application built with **FastAPI**, **PyTorch**, and a **T5 Transformer model**. Enter text or a conversation in a simple browser interface and generate a concise summary using a locally stored model.

## Features

- Generate summaries from user-provided text or dialogue.
- Web interface built with HTML, CSS, and JavaScript.
- REST API powered by FastAPI.
- Text preprocessing (whitespace normalization, HTML tag removal, and lowercasing).
- Model inference on Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU, depending on availability.

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Machine learning:** Hugging Face Transformers, PyTorch, T5
- **Frontend:** HTML, CSS, JavaScript
- **Templates:** Jinja2

## Project Structure

```text
TEXTSUMMARIZERAPP/
├── app.py                     # FastAPI application and summarization logic
├── index.html                 # Web interface
├── saved_summary_model/       # Local model and tokenizer files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── .gitattributes              # Git LFS configuration (if tracked)
├── .gitignore                  # Ignored local files (if created)
└── README.md
```

## Getting Started

### 1. Clone the repository

The model weights are large and should be tracked with **Git LFS**. Install [Git LFS](https://git-lfs.com/) before cloning, then run:

```bash
git lfs install
git clone https://github.com/adibWd/Text-Summarizer-App.git
cd Text-Summarizer-App
```

If the repository has already been cloned and the model file is only an LFS pointer, run:

```bash
git lfs pull
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate` instead.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install fastapi uvicorn transformers torch sentencepiece jinja2
```

> **Note:** The application loads both the model and tokenizer from `./saved_summary_model`. Make sure the downloaded folder contains the model weights and tokenizer files before starting the server. Compatibility can depend on your Python and package versions.

### 4. Run the app

From the project root:

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in a browser.

The interactive FastAPI API documentation is available at **http://127.0.0.1:8000/docs**.

## API Usage

### `POST /summarize/`

**Request body**

```json
{
  "dialogue": "Alice: Are we meeting at 3 PM? Bob: Yes, let's meet in the library."
}
```

**Response format**

```json
{
  "summary": "<model-generated summary>"
}
```

**Example with cURL**

```bash
curl -X POST "http://127.0.0.1:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{"dialogue":"Alice: Are we meeting at 3 PM? Bob: Yes, let us meet in the library."}'
```

## How It Works

1. The user enters text in the web interface.
2. The frontend sends a JSON request to `POST /summarize/`.
3. The backend cleans the text and tokenizes it (maximum input length: 512 tokens).
4. The local T5 model generates a summary (maximum output length: 150 tokens, 4-beam search).
5. The API returns the summary as JSON for display in the browser.

## Notes

- This project uses a **local model**, rather than a hosted inference API.
- `model.safetensors` is larger than GitHub's regular 100 MB per-file Git limit; use **Git LFS** when committing it.
- Input longer than 512 tokens is truncated by the current implementation.
- Model-generated summaries may omit details or contain inaccuracies; review important outputs before relying on them.

## Repository

[GitHub — Text-Summarizer-App](https://github.com/adibWd/Text-Summarizer-App)
