from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from operator import itemgetter
import os
import sys



class ChatBot:
    def __init__(self, pdf, model, db='db'):
        self.PDF = pdf
        self.MODEL_NAME = model
        self.FAISS_DB_NAME = db
        self.model = Ollama(model=self.MODEL_NAME)
        self.embedding = OllamaEmbeddings(model=self.MODEL_NAME)

        self.chain = (
            {"context": itemgetter("question") | self.generate_vectorstore_and_retriever(),
                "question": itemgetter("question"),
                }
            | self.promptTemplate()
            | self.model
            | self.stringParser()
            )

    def stringParser(self):
        return StrOutputParser()
    
    def promptTemplate(self):
        template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "I don't know", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        self.prompt = PromptTemplate.from_template(template)
        self.prompt.format(context="Here is some context", question="Here is a question")

        return self.prompt
    
    def generate_vectorstore_and_retriever(self):
        if self.FAISS_DB_NAME not in os.listdir():
            loader = PyPDFLoader(self.PDF)
            pages = loader.load_and_split()
            vectorstore = FAISS.from_documents(documents=pages, embedding=self.embedding)
            vectorstore.save_local(self.FAISS_DB_NAME)
        else:
            vectorstore = FAISS.load_local(self.FAISS_DB_NAME, embeddings=self.embedding, allow_dangerous_deserialization=True)

        self.retriever = vectorstore.as_retriever()
        return self.retriever
    
    
    def askQuestion(self,question):
        response = self.chain.invoke({"question": question})
        return response
    

    def __str__(self):
        return f"{self.PDF}/{self.MODEL_NAME}/{self.FAISS_DB_NAME}"
    

if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        bot = ChatBot(*args)
        print(f"Vectorstore database stored successfull in {os.path.abspath(args[2])}")
    except Exception as e:
        print(e)
        print(f"Error: Run either Ollama server by running `ollama serve` in your terminal if you didn't started or `ollama pull {args[1]}` if you didn't installed {args[1]} model in your local machine")
