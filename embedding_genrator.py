from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import sys


file = TextLoader(sys.argv[1] if len(sys.argv)>1 else "faq.txt")
raw_content = file.load()[0].page_content.splitlines()
parsed_docs = []
current_entry = {}
is_answer = False
embedding = OllamaEmbeddings(model="mxbai-embed-large")
CHROMA_DB_NAME = sys.argv[2] if len(sys.argv)>2 else "db"
documents = []

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
    

for doc in parsed_docs:
    documents.append(Document(
        page_content=f"Question: {doc["question"]}\nAnswer: {doc["answer"]}\nReferences: {doc.get("references", "")}",
        metadata={
            "id": doc["id"],
            "question": doc["question"],
            "context": doc["context"],
            "references": doc.get("references", ""),
        }
    ))


vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=CHROMA_DB_NAME
)
print("Successfully genrated")