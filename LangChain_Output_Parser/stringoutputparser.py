from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

huggingface_llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=huggingface_llm)

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

report_prompt = report_template.invoke({'topic':'black hole'})

detailed_report = chat_model.invoke(report_prompt)

summary_prompt = summary_template.invoke({'text':detailed_report.content})

summary_result = chat_model.invoke(summary_prompt)

print(summary_result.content)
