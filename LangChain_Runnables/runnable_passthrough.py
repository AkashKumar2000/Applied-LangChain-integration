# 1. Imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()

# 3. Define structured output schema
class Review(BaseModel):
    points: str = Field(description="Key points about the topic")

# 4. Define prompts
prompt1 = PromptTemplate(
    template="Write bad things about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following topic - {text}",
    input_variables=["text"]
)

# 5. Setup model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
structured_model = model.with_structured_output(Review)

# 6. Build chains
joke_gen_chain = prompt1 | structured_model

parallel_chain = RunnableParallel({
    "original": RunnablePassthrough(),
    "explanation": prompt2 | structured_model
})

final_chain = joke_gen_chain | parallel_chain

# 7. Run chain
result = final_chain.invoke({"topic": "social media"})
print("ORIGINAL:", result["original"].points)
print("EXPLANATION:", result["explanation"].points)


# WHAT IS ACTUALLY HAPPENING:
#
# Step 1 - final_chain.invoke({"topic": "social media"}) is called.
#
# Step 2 - joke_gen_chain runs first (sequential):
#          prompt1 builds → "Write bad things about social media"
#          structured_model sends it to Mistral-7B and returns:
#          Review(points="...bad things about social media...")
#
# Step 3 - The Review object is passed into parallel_chain.
#          RunnableParallel splits it into TWO branches simultaneously:
#
#          Branch 1 - "original":
#          RunnablePassthrough() does NOTHING — it just passes the
#          Review object through unchanged.
#          Result: Review(points="...bad things about social media...")
#
#          Branch 2 - "explanation":
#          prompt2 builds → "Explain the following topic - ...bad things..."
#          structured_model sends it to Mistral-7B and returns:
#          Review(points="...explanation of the bad things...")
#
# Step 4 - RunnableParallel collects both results into a dict:
#          {
#              "original":    Review(points="...bad things..."),
#              "explanation": Review(points="...explanation...")
#          }
#
# Step 5 - We access the results:
#          result["original"].points    -> original bad things about social media
#          result["explanation"].points -> explanation of those bad things
#
# KEY CONCEPT - RunnablePassthrough:
#   It is used when you want to KEEP the original output of a previous step
#   and also process it further in parallel. Without it, the original Review
#   object would be lost after the explanation chain runs.

