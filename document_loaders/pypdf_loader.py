from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# input paths and variables
text_path = 'document_loaders/books/ather_report.pdf'

# loading access keys
load_dotenv()

# instantiate models
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    provider="auto",  # let Hugging Face choose the best provider for you
)
model = ChatHuggingFace(llm=llm)

# instantiate string parser
parser = StrOutputParser()

# load document
loader = PyMuPDFLoader(text_path)
docs = loader.load()
# print content and metadata
print(len(docs))
print(docs[0].metadata)
print('--'*30)
print(docs[-10].page_content)

# question-answer from pdf
prompt = PromptTemplate.from_template("Answer the following question based on the text as follows \n {text}: \n question: {question}")

chain = prompt | model | parser

# invoking our chain
response = chain.invoke({"text": docs[0].page_content, "question": "What are the topics?"})
print(response)

