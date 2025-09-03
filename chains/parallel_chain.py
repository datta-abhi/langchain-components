from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel

# fetching env variables for authentication
load_dotenv()

# reading a text file using context manager
with open("input_files/india_us_tariff.txt", "r",  encoding='utf-8') as file_object:
    text = file_object.read()
     
# instantiate models
model1 = ChatOllama(model="llama3.2")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")   

# create prompts

# prompt for creating notes
prompt1 = PromptTemplate.from_template("Generate short bullet point notes based on following text: \n {text}")
# prompt for creating quiz questions
prompt2 = PromptTemplate.from_template("Generate 3 questions to test a candidates understanding of following text: \n {text}")
# prompt for merging notes and quiz
prompt3 = PromptTemplate.from_template("Merge the provided notes and quiz into a single document \n notes: \n {notes} \n quiz: \n {quiz}")

# initialise parser
parser = StrOutputParser()

# chaining in parallel
parallel_chain = RunnableParallel({'notes': prompt1 | model1 | parser,
                                   'quiz': prompt2 | model2 | parser})

merge_chain = prompt3 | model1 | parser

outer_chain = parallel_chain | merge_chain

# get response
response = outer_chain.invoke({'text': text}) 
print(response)
