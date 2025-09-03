from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# topic
topic= "dragons"

# instantiate model
model = ChatOllama(model = "llama3.2")

# create prompt
prompt = PromptTemplate.from_template("Generate a 100-200 word story about {topic}")

# initialise a string output parser
parser = StrOutputParser()

# create chain
chain = prompt | model | parser

# get response
response = chain.invoke({"topic": topic})
print(response)