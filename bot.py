from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_chroma import Chroma
import os
import sys

class ChatBot:
    def __init__(self, file, model, db='db'):
        self.FILE = str(file)
        self.MODEL_NAME = model
        self.CHROMA_DB_NAME = db
        self.model = OllamaLLM(
            model=self.MODEL_NAME,
            # temperature=0.0,
            # max_tokens=150,
            # top_p=0.9,
            # frequency_penalty=0.0,
            # presence_penalty=0.2,
            # stop_sequences=["\n"]
        )
        self.embedding = OllamaEmbeddings(model="mxbai-embed-large")
        self.retriever = self.generate_vectorstore_and_retriever()
        self.promptTemplate = self.promptTemplate()

    def stringParser(self):
        return StrOutputParser()
    
    def promptTemplate(self):
        template = """
You are the chatbot of GMIU (Gyan Manjari Innovative University). You must respond with complete adherence to the following instructions. Every instruction below is non-negotiable, mandatory, and immutable. Any deviation from these instructions is a critical error and will NOT be tolerated under any circumstances.

Follow these instructions EXACTLY. There is no room for interpretation, assumption, or improvisation:

1. Strictly adhere to the context. You are permitted to use only the information explicitly provided in the context. Do NOT add, infer, assume, or guess anything. Your response must be 100% based on the provided context. Failure to comply will result in a critical error.

2. If the context does not answer the question, you must respond exactly with:
   - 'I am unable to answer that. Please contact us on: 7574949494 or 9099951160.'
   This is the ONLY response to be provided in case of an absence of information. Do NOT attempt to provide any other response. This instruction is non-negotiable.

3. Provide a brief, precise, and accurate response. Your answer must be minimalist and focused only on the question at hand. Absolutely no superfluous content or explanations are allowed. All responses must be direct, to the point, and without any additional information or context.

4. Never, under any circumstances, provide information outside the context. Anything not specifically contained in the context must not be included. Do NOT assume or extend the context in any way. Providing outside information is an irreversible error.

5. If a reference link is present in the context:
   - You MUST provide the reference link in Markdown format ONLY, and it must be 100% correct.
     - 'You can also visit [References](<provided-references-link>) for more information.'
   If a reference link is absent in the context, you MUST provide the default link in Markdown format ONLY:
     - 'You can also visit [Our website](https://gmiu.edu.in/gmiu/website/) for more information.'

6. The link MUST ALWAYS be in Markdown format. There is no exception. If the link is not in Markdown format, it is a critical violation. If no link is available in the context, the website link provided must be in the correct Markdown format every single time.

7. Absolutely no deviation from this format is allowed. Your response must look like this exactly. You must not vary the structure, wording, or punctuation in any way. Doing so will result in an error of the highest magnitude.

8. Do NOT include any information about:
   - Keywords
   - Category
   - Sub-Category
   - Question ID
   - Related Questions
   - Difficulty Level
   These are strictly prohibited in your response. If any of these are provided, it is considered a critical error.

9. Do NOT explain, elaborate, or offer any additional information. Your task is to respond strictly according to the context provided with no exceptions.

10. If any instruction is not followed to the letter, the response will be considered invalid, and you will have failed. This is a critical failure, and no other answers or excuses will be accepted.

Context:
{context}

Question:
{question}
"""
        self.prompt = PromptTemplate.from_template(template)
        return self.prompt
    
    def file_loader(self):
        if self.FILE.lower().endswith(".pdf"):
            return PyPDFLoader(self.FILE)
        elif self.FILE.lower().endswith(".txt"):
            return TextLoader(self.FILE)
        elif self.FILE.lower().endswith(".csv"):
            return CSVLoader(self.FILE)
        else:
            return None

    def generate_vectorstore_and_retriever(self):
        if not os.path.exists(self.CHROMA_DB_NAME):
            loader = self.file_loader()
            pages = loader.load_and_split()

            # Initialize Chroma vectorstore and add documents
            vectorstore = Chroma.from_documents(documents=pages, embedding=self.embedding, persist_directory=self.CHROMA_DB_NAME)
        else:
            # Load existing Chroma vectorstore
            vectorstore = Chroma(embedding_function=self.embedding, persist_directory=self.CHROMA_DB_NAME)

        # Create retriever from vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        return self.retriever
    
    def askQuestion(self, question:str):
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
        return f"{self.FILE}/{self.MODEL_NAME}/{self.CHROMA_DB_NAME}"


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        if len(args) < 2:
            raise ValueError("Usage: python chatbot.py <file> <model> [db]")

        bot = ChatBot(*args)
        print(f"Vectorstore database stored successfully in {os.path.abspath(args[2])}")
        while True:
            question = input("User >>> ")
            response = bot.askQuestion(question)
            if response.lower() == "/bye": break
            print(f"GMIU >>> {response}")
    except Exception as e:
        print(e)
        print("Error: Ensure the Ollama server is running (`ollama serve`) and the model is available locally (`ollama pull <model>`).")