from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

# input review feedback
review = "This smartphone comes with a wonderful design and form factor"

# fetching env variables for authentication
load_dotenv()

# initialize model
model_classify = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model_reply = ChatOllama(model="llama3.2")

# initialize parser
parser =StrOutputParser()

# classify sentiment from product feedback string
class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="Classify the given feedback as positive or negative")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_classify = PromptTemplate.from_template("Classify the following text in this output format {format_instructions}: \n {text}")
partial_prompt_classify = prompt_classify.partial(format_instructions = parser2.get_format_instructions())

classify_chain = partial_prompt_classify | model_classify | parser2 

# sentiment = chain_classify.invoke({'text': review}).sentiment
# print(sentiment)

# Branching conditionally

# create prompts
prompt_pos = PromptTemplate.from_template("Write a short grateful acknowledgement response to this positive feedback \n {feedback}. Give only the response")

prompt_neg = PromptTemplate.from_template("Write a short apologetic response to this negative feedback \n {feedback}. Give only the response")

condition1 = lambda x: x.sentiment == 'positive'
chain1 = prompt_pos | model_reply | parser

condition2 = lambda x: x.sentiment == 'negative'
chain2 = prompt_neg | model_reply | parser

default_chain = lambda x: "could not classify sentiment"

branch_chain = RunnableBranch((condition1,chain1),
                              (condition2,chain2),
                              RunnableLambda(default_chain))

final_chain = classify_chain | branch_chain

final_response = final_chain.invoke({'text': review})
print(final_response)
