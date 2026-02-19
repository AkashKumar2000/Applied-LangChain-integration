from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.messages import SystemMessage  ,HumanMessage , AIMessage

import os 

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token= hf_token
)

model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage("You are a poetry expert")
]

while True:
    user_input= input("you:")
    chat_history.append(HumanMessage(user_input))
    if user_input=="exit":
        break
    result= model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI:", result.content)


print(chat_history)