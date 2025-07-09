import os
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv('BASE_URL', "127.0.0.1:11434")
CHROMA_DB_NAME = os.getenv('CHROMA_DB_NAME', "db")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "mxbai-embed-large")
TEXT_FOLDER = os.getenv("TEXT_FOLDER", None)
FILES = [f"{TEXT_FOLDER}/{file}" for file in os.listdir(TEXT_FOLDER)] if TEXT_FOLDER else eval(os.getenv('FILES', "[]"))

embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)
documents = []

for file_path in FILES:
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping.")
        continue

    file_loader = TextLoader(file_path)
    raw_content = file_loader.load()[0].page_content.splitlines()
    parsed_docs = []
    current_entry = {}
    is_answer = False

    for line in raw_content:
        line = line.strip()
        if not line:
            continue
        try:
            if line.lower().startswith("question id"):
                if current_entry:
                    parsed_docs.append(current_entry)
                    current_entry = {}
                current_entry["id"] = line.split(":", 1)[1].strip()
                is_answer = False

            elif line.lower().startswith("question"):
                current_entry["question"] = line.split(":", 1)[1].strip()
                is_answer = False

            elif line.lower().startswith("answer"):
                is_answer = True
                current_entry["answer"] = line.split(":", 1)[1].strip()

            elif line.lower().startswith("context"):
                current_entry["context"] = line.split(":", 1)[1].strip()
                is_answer = False

            elif line.lower().startswith("references"):
                current_entry["references"] = line.split(":", 1)[1].strip()
                is_answer = False

            elif line.lower().startswith("keyword") or line.lower().startswith("difficulty level") or line.lower().startswith("permissions") or line.lower().startswith("time sensitivity") or line.lower().startswith("related questions") or line.lower().startswith("-"):
                is_answer = False

            elif is_answer:
                current_entry["answer"] += "\n" + line.strip()

        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Exception: {e}")

    if current_entry:
        parsed_docs.append(current_entry)

    if not parsed_docs:
        print(f"No valid entries found in the file: {file_path}. Skipping.")
        continue

    for doc in parsed_docs:
        if not "answer" in doc:
            print(f"Skipping entry with missing answer: {doc['id']}, {doc['question']}")
            continue
        documents.append(Document(
            page_content=f"Question: {doc['question']}\nAnswer: {doc['answer']}\nReferences: {doc.get('references', '')}",
            metadata={
                "id": doc["id"],
                "question": doc["question"],
                "context": doc["context"],
                "references": doc.get("references", ""),
            }
        ))

if documents:
    Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_DB_NAME
    )
    print("Vectorstore Generated Successfully!")
else:
    print("No documents to store. Exiting.")