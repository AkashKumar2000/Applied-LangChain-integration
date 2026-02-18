from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token=hf_token
)

text = "My name is Akash, I need to vectorize it."

result = embeddings.embed_query(text)

print(result)
print(len(result))
