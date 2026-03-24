from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

chat_model = ChatOpenAI()

output_parser = StrOutputParser()

fact_chain = prompt_template | chat_model | output_parser

response = fact_chain.invoke({'topic':'cricket'})

print(response)

fact_chain.get_graph().print_ascii()
