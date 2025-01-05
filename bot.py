from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv('BASE_URL', "127.0.0.1:11434")
PROMPT_FILE = os.getenv('PROMPT_FILE', "prompt.txt")
CHROMA_DB_NAME = os.getenv('CHROMA_DB_NAME', "db")
MODEL = os.getenv('MODEL', "llama3.2:3b")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "mxbai-embed-large")
FILES = eval(os.getenv('FILES', "[]"))

class ChatBot:
    def __init__(self):
        self.FILES = [file.strip() for file in FILES if file.strip()]
        self.MODEL_NAME = MODEL
        self.CHROMA_DB_NAME = CHROMA_DB_NAME
        self.model = OllamaLLM(
            model=self.MODEL_NAME,
            base_url=BASE_URL
        )
        self.embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)
        self.retriever = self.generate_vectorstore_and_retriever()
        self.promptTemplate = self.promptTemplate()

    def stringParser(self):
        return StrOutputParser()
    
    def promptTemplate(self):
        if not os.path.exists(PROMPT_FILE):
            print(f"Prompt file {PROMPT_FILE} not found.")
            return None
        
        with open(PROMPT_FILE) as f:
            template = f.read()
        self.prompt = PromptTemplate.from_template(template)
        return self.prompt

    def file_loader(self, file_path):
        if file_path.lower().endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif file_path.lower().endswith(".txt"):
            return TextLoader(file_path)
        elif file_path.lower().endswith(".csv"):
            return CSVLoader(file_path)
        else:
            return None

    def generate_vectorstore_and_retriever(self):
        documents = []
        if not os.path.exists(self.CHROMA_DB_NAME):
            for file in self.FILES:
                loader = self.file_loader(file)
                if loader:
                    documents.extend(loader.load_and_split())
                else:
                    print(f"Unsupported file format for {file}")
            
            vectorstore = Chroma.from_documents(documents=documents, embedding=self.embedding, persist_directory=self.CHROMA_DB_NAME)
        else:
            vectorstore = Chroma(embedding_function=self.embedding, persist_directory=self.CHROMA_DB_NAME)

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return self.retriever

    def askQuestion(self, question: str):
        if question.lower().startswith("dev@"):
            context_docs = self.retriever.invoke(question.split("dev@")[1])
            relevent_docs = "\n".join([doc.page_content for doc in context_docs])
            return relevent_docs
        context_docs = self.retriever.invoke(question)
        relevent_docs = "\n".join([doc.page_content for doc in context_docs])
        prompt = self.promptTemplate.format(context=relevent_docs, question=question)
        response = self.model.invoke(prompt)
        return response.strip()
    
    def __str__(self):
        return f"{self.FILES}/{self.MODEL_NAME}/{self.CHROMA_DB_NAME}"


if __name__ == "__main__":
    try:
        bot = ChatBot()
        print(f"Vectorstore database stored successfully in folder {CHROMA_DB_NAME}")
        while True:
            question = input("User >>> ")
            if question.lower() == "/bye": break
            response = bot.askQuestion(question)
            print(f"GMIU >>> {response}")
    except Exception as e:
        print(e)
        print("Error: Ensure the Ollama server is running (`ollama serve`) and the model is available locally (`ollama pull <model>`).")