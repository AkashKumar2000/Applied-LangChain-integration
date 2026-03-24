from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
huggingface_llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=huggingface_llm)

json_parser = JsonOutputParser()

facts_template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': json_parser.get_format_instructions()}
)

facts_chain = facts_template | chat_model | json_parser

parsed_result = facts_chain.invoke({'topic':'black hole'})

print(parsed_result)
