from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# topic
topic= "India- Russia relationships"

# instantiate model
model = ChatOllama(model = "llama3.2")
# initialise a string output parser
parser = StrOutputParser()

# 1st step: report generator
prompt1 = PromptTemplate.from_template("Generate a 300 word report about {topic}")

# 2nd step: summary generator
prompt2 = PromptTemplate.from_template("Generate a concise summary from the following text. \n {text}")

# create chain
chain = prompt1 | model | parser | prompt2 | model | parser

# get response
response = chain.invoke({"topic": topic})
print(response)
print('--'*50)
print(prompt2)