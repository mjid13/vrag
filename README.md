# VRAG

VRAG is a small Flask web app for chatting with company documents through OpenAI file search. It creates an OpenAI vector store, uploads local documents into it, creates an assistant wired to that store, and exposes a dark UI for document upload, deletion, and question answering.

The app is useful for a local, lightweight retrieval-augmented generation workflow: drop in files, let them index, then ask questions in English or Arabic.

## Features

- Chat UI at `/` for asking questions about indexed documents.
- Document manager at `/docs` with drag-and-drop upload, refresh, indexing status, and delete actions.
- Automatic startup sync from the local `company_docs/` folder.
- Filesystem watcher that uploads newly created or modified documents while the server is running.
- Source citation display when OpenAI file search returns file citations.
- Supported document types: PDF, TXT, Markdown, DOCX, HTML, and JSON.

## Requirements

- Python 3.10 or newer recommended.
- An OpenAI API key with access to assistants, vector stores, and file search.

Python dependencies are listed in `requirements.txt`:

```txt
flask
openai
watchdog
python-dotenv
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The app also creates local runtime files and folders:

- `.vrag_config.json` stores the OpenAI assistant ID and vector store ID.
- `company_docs/` stores local uploaded documents.

Both are ignored by git.

## Run

Start the server:

```bash
python app.py
```

Open the app:

- Chat: `http://localhost:5000/`
- Documents: `http://localhost:5000/docs`

On startup, the app will:

1. Load `.env`.
2. Create or reuse the OpenAI vector store saved in `.vrag_config.json`.
3. Create or reuse the assistant saved in `.vrag_config.json`.
4. Create `company_docs/` if it does not exist.
5. Upload supported files already present in `company_docs/`.
6. Watch `company_docs/` for file changes.

## Usage

Use `/docs` to upload documents through the browser, or place files directly in `company_docs/` while the server is running. Supported files are uploaded to OpenAI and attached to the configured vector store.

Then use `/` to ask questions. The assistant is instructed to search uploaded files before answering, cite source documents when possible, and clearly say when an answer is not present in the documents.

## API Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Render the chat page. |
| `/docs` | GET | Render the document manager. |
| `/docs/list` | GET | Return local document names, sizes, support status, and indexed status. |
| `/docs/upload` | POST | Upload one document from a multipart `file` field. |
| `/docs/delete` | POST | Delete a local document and remove it from OpenAI storage. |
| `/chat` | POST | Send a message to the assistant and return the response plus citations. |

Example chat request:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize the uploaded documents"}'
```

## Project Structure

```text
.
|-- app.py                 # Flask app, OpenAI setup, document sync, chat endpoint
|-- requirements.txt       # Python dependencies
|-- templates/
|   |-- chat.html          # Chat UI
|   `-- docs.html          # Document upload and management UI
|-- company_docs/          # Local document storage, created at runtime
|-- .vrag_config.json      # Runtime OpenAI IDs, created at runtime
`-- .env                   # Local secrets
```

## Configuration

The main configuration is environment-based:

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key used by the OpenAI Python SDK. |

Runtime constants are defined in `app.py`:

- `DOCS_DIR`: local document folder, currently `company_docs`.
- `SUPPORTED_EXTENSIONS`: `.pdf`, `.txt`, `.md`, `.docx`, `.html`, `.json`.
- Assistant model: `gpt-4o-mini`.
- Server port: `5000`.

## Troubleshooting

If startup fails with an authentication error, check that `.env` exists and contains a valid `OPENAI_API_KEY`.

If files appear as not indexed, refresh `/docs` after a few seconds. Large files can take longer to upload and index.

If the assistant or vector store was deleted in OpenAI, remove `.vrag_config.json` and restart the app. A new assistant and vector store will be created.

If direct file edits are not detected, make sure the server is still running and that files are placed directly inside `company_docs/`, not in nested folders. The watcher is not recursive.
