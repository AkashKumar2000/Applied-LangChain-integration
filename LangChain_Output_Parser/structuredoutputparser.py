from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Define the model
huggingface_llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=huggingface_llm)

response_schemas = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

facts_template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':structured_parser.get_format_instructions()}
)

facts_chain = facts_template | chat_model | structured_parser

parsed_result = facts_chain.invoke({'topic':'black hole'})

print(parsed_result)
