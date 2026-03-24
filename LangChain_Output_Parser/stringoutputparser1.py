from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


chat_model = ChatOpenAI()

# 1st prompt -> detailed report
report_template = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
summary_template = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

string_parser = StrOutputParser()

pipeline_chain = report_template | chat_model | string_parser | summary_template | chat_model | string_parser

final_result = pipeline_chain.invoke({'topic':'black hole'})

print(final_result)
