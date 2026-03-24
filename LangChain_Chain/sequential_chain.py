from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

report_prompt = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

summary_prompt = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chat_model = ChatOpenAI()

output_parser = StrOutputParser()

sequential_chain = report_prompt | chat_model | output_parser | summary_prompt | chat_model | output_parser

response = sequential_chain.invoke({'topic': 'Unemployment in India'})

print(response)

sequential_chain.get_graph().print_ascii()
