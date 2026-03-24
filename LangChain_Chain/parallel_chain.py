from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

chat_model = ChatOpenAI()

str_parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

sentiment_prompt = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_parser.get_format_instructions()}
)

sentiment_classifier_chain = sentiment_prompt | chat_model | pydantic_parser

positive_response_prompt = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

negative_response_prompt = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

response_branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', positive_response_prompt | chat_model | str_parser),
    (lambda x:x.sentiment == 'negative', negative_response_prompt | chat_model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

feedback_chain = sentiment_classifier_chain | response_branch_chain

print(feedback_chain.invoke({'feedback': 'This is a beautiful phone'}))

feedback_chain.get_graph().print_ascii()
