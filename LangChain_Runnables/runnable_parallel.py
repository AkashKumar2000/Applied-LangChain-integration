# 1. Imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()

# 3. Define structured output schema (common for both prompts)
class Review(BaseModel):
    points: str = Field(description="The key points about the topic")

# 4. Define prompts
prompt1 = PromptTemplate(
    template="Tell me bad things about {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Tell me good things about {text}",
    input_variables=['text']
)

# 5. Setup model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
structured_model = model.with_structured_output(Review)

# 6. Build parallel chain
parallel_chain = RunnableParallel({
    "bad_things":  prompt1 | structured_model,
    "good_things": prompt2 | structured_model
})

# 7. Run chain
result = parallel_chain.invoke({"text": "social media"})
print("BAD THINGS:", result["bad_things"].points)
print("GOOD THINGS:", result["good_things"].points)


# WHAT IS ACTUALLY HAPPENING:
#
# Step 1 - parallel_chain.invoke({"text": "social media"}) is called.
#          The same input {"text": "social media"} is sent to BOTH chains at the same time.
#
# Step 2 - Both chains run in PARALLEL (simultaneously):
#
#          Chain 1 (bad_things):
#          prompt1 builds → "Tell me bad things about social media"
#          structured_model sends it to Mistral-7B and returns:
#          Review(points="...bad points about social media...")
#
#          Chain 2 (good_things):
#          prompt2 builds → "Tell me good things about social media"
#          structured_model sends it to Mistral-7B and returns:
#          Review(points="...good points about social media...")
#
# Step 3 - RunnableParallel collects both results into a dict:
#          {
#              "bad_things":  Review(points="..."),
#              "good_things": Review(points="...")
#          }
#
# Step 4 - We access the results using:
#          result["bad_things"].points  -> bad points about social media
#          result["good_things"].points -> good points about social media
#
# NOTE: Both prompt1 and prompt2 share the same structured_model (Review schema).
#       This means both outputs have the same structure — just different content.

