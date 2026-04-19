import os
import json
import time
import threading
import logging

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".vrag_config.json")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "company_docs")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html", ".json"}

# Maps local filename -> {"file_id": ..., "vs_file_id": ...}
file_registry = {}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_or_create_vector_store(config):
    vs_id = config.get("vector_store_id")
    if vs_id:
        try:
            client.vector_stores.retrieve(vs_id)
            logger.info(f"Using existing vector store: {vs_id}")
            return vs_id
        except Exception:
            logger.info("Stored vector store not found, creating new one")

    vs = client.vector_stores.create(name="vrag-company-docs")
    config["vector_store_id"] = vs.id
    save_config(config)
    logger.info(f"Created vector store: {vs.id}")
    return vs.id


def get_or_create_assistant(config, vector_store_id):
    asst_id = config.get("assistant_id")
    if asst_id:
        try:
            client.beta.assistants.retrieve(asst_id)
            logger.info(f"Using existing assistant: {asst_id}")
            return asst_id
        except Exception:
            logger.info("Stored assistant not found, creating new one")

    assistant = client.beta.assistants.create(
        name="VRAG Document Assistant",
        instructions=(
            "You are a helpful assistant that answers questions based on the uploaded company documents. "
            "Always search the files for relevant information before answering. "
            "Cite the source document when possible. "
            "If the answer is not in the documents, say so clearly. "
            "You can respond in Arabic or English depending on the user's language."
        ),
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
    config["assistant_id"] = assistant.id
    save_config(config)
    logger.info(f"Created assistant: {assistant.id}")
    return assistant.id


def upload_file(filepath):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.info(f"Skipping unsupported file: {filename}")
        return

    config = load_config()
    vs_id = config.get("vector_store_id")
    if not vs_id:
        return

    # Remove old version if exists
    if filename in file_registry:
        remove_file(filename)

    try:
        with open(filepath, "rb") as f:
            uploaded = client.files.create(file=f, purpose="assistants")

        vs_file = client.vector_stores.files.create(
            vector_store_id=vs_id, file_id=uploaded.id
        )

        file_registry[filename] = {
            "file_id": uploaded.id,
            "vs_file_id": vs_file.id,
        }
        logger.info(f"Uploaded and indexed: {filename}")
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}")


def remove_file(filename):
    config = load_config()
    vs_id = config.get("vector_store_id")
    entry = file_registry.pop(filename, None)
    if not entry or not vs_id:
        return

    try:
        client.vector_stores.files.delete(
            vector_store_id=vs_id, file_id=entry["file_id"]
        )
    except Exception:
        pass

    try:
        client.files.delete(entry["file_id"])
    except Exception:
        pass

    logger.info(f"Removed from vector store: {filename}")


def sync_existing_files():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        return

    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                upload_file(filepath)


class DocsEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        time.sleep(0.5)  # wait for file write to finish
        upload_file(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        time.sleep(0.5)
        upload_file(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        filename = os.path.basename(event.src_path)
        remove_file(filename)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/docs")
def docs_page():
    return render_template("docs.html")


@app.route("/docs/list")
def docs_list():
    files = []
    if os.path.exists(DOCS_DIR):
        for filename in sorted(os.listdir(DOCS_DIR)):
            filepath = os.path.join(DOCS_DIR, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                stat = os.stat(filepath)
                size_kb = round(stat.st_size / 1024, 1)
                files.append({
                    "name": filename,
                    "size": f"{size_kb} KB",
                    "supported": ext in SUPPORTED_EXTENSIONS,
                    "indexed": filename in file_registry,
                })
    return jsonify({"files": files})


@app.route("/docs/upload", methods=["POST"])
def docs_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    filepath = os.path.join(DOCS_DIR, filename)
    f.save(filepath)
    # Watchdog will pick it up, but also upload directly for immediate feedback
    upload_file(filepath)
    return jsonify({"success": True, "name": filename})


@app.route("/docs/delete", methods=["POST"])
def docs_delete():
    data = request.get_json()
    filename = data.get("name", "")
    if not filename:
        return jsonify({"error": "No filename"}), 400
    filename = secure_filename(filename)
    filepath = os.path.join(DOCS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        remove_file(filename)
    return jsonify({"success": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    config = load_config()
    assistant_id = config.get("assistant_id")
    if not assistant_id:
        return jsonify({"error": "Assistant not initialized"}), 500

    try:
        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_message
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant_id
        )

        if run.status != "completed":
            return jsonify({"error": f"Run failed: {run.status}"}), 500

        messages = client.beta.threads.messages.list(
            thread_id=thread.id, order="desc", limit=1
        )

        response_text = ""
        citations = []

        for msg in messages.data:
            if msg.role == "assistant":
                for block in msg.content:
                    if block.type == "text":
                        response_text = block.text.value
                        if block.text.annotations:
                            for ann in block.text.annotations:
                                if ann.type == "file_citation":
                                    try:
                                        cited_file = client.files.retrieve(
                                            ann.file_citation.file_id
                                        )
                                        citations.append(cited_file.filename)
                                    except Exception:
                                        pass
                                    response_text = response_text.replace(
                                        ann.text, ""
                                    )
                break

        return jsonify({
            "response": response_text.strip(),
            "citations": list(set(citations)),
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    config = load_config()
    vs_id = get_or_create_vector_store(config)
    assistant_id = get_or_create_assistant(config, vs_id)

    logger.info("Syncing existing documents...")
    sync_existing_files()

    observer = Observer()
    observer.schedule(DocsEventHandler(), DOCS_DIR, recursive=False)
    observer.start()
    logger.info(f"Watching {DOCS_DIR} for changes")

    try:
        app.run(debug=False, port=5000)
    finally:
        observer.stop()
        observer.join()
