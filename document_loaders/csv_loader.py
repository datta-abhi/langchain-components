from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# input paths and variables
text_path = 'document_loaders/sample_inputs/Social_Network_Ads.csv'

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
loader = CSVLoader(text_path)
docs = loader.load()
# print content and metadata
print(len(docs))
print(docs[-1].metadata)
print('--'*30)
print(docs[0].page_content)