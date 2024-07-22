from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

class PDFChatbot:
    def __init__(self, pdf_path, faiss_index_path="faiss_index", google_api_key=None):
        # Load environment variables
        load_dotenv()
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        
        # Initialize embeddings and model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.pdf_path = pdf_path
        self.faiss_index_path = faiss_index_path

        # Define the prompt template
        self.prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
        the provided context, just say, "Sorry, I don't know about that. Please ask a different question or call us on: +919099951160; +917574949494." Don't provide the wrong answer.

        Context:\n {context}\n
        Question:\n {question}\n

        Answer:
        """

    def get_pdf_text(self):
        """Extract text from a PDF file."""
        text = ""
        pdf = PdfReader(self.pdf_path)
        for page in pdf.pages:
            text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        """Split text into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        """Get or create a vector store."""
        if os.path.exists(self.faiss_index_path + ".index"):
            vector_store = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            vector_store.save_local(self.faiss_index_path)
        return vector_store

    def get_conversational_chain(self):
        """Create the question-answering chain."""
        prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(self.model, chain_type="stuff", prompt=prompt)
        return chain

    def user_input(self, user_question):
        """Handle user input and generate responses."""
        vector_store = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        
        chain = self.get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]

    def get_answer(self, user_question):
        """Get an answer to the user question from the PDF."""
        raw_text = self.get_pdf_text()
        text_chunks = self.get_text_chunks(raw_text)
        self.get_vector_store(text_chunks)
        result = self.user_input(user_question)
        return result