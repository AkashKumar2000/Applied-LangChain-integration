# 1. Imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()

# 3. Define structured output schemas
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

class Explanation(BaseModel):
    explanation: str = Field(description="The explanation of the joke")
    funny_because: str = Field(description="Why the joke is funny")

# 4. Define prompts
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=['text']
)

# 5. Setup model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
structured_model = model.with_structured_output(Joke)
structured_model2 = model.with_structured_output(Explanation)

# 6. Converter: Joke object -> dict for prompt2
joke_to_text = RunnableLambda(lambda joke: {"text": f"{joke.setup} - {joke.punchline}"})

# 7. Build and run chain
chain = prompt1 | structured_model | joke_to_text | prompt2 | structured_model2

result = chain.invoke({"topic": "cats"})
print(result.explanation)
print(result.funny_because)


# WHAT IS ACTUALLY HAPPENING:
#
# Step 1 - prompt1 receives {"topic": "cats"} and builds the text:
#          "Write a joke about cats"
#
# Step 2 - structured_model sends that text to Mistral-7B and forces the output
#          to match the Joke schema, returning:
#          Joke(setup="Why don't cats play poker?", punchline="Too many cheetahs!")
#
# Step 3 - joke_to_text is a RunnableLambda (a simple converter function) that
#          takes the Joke object and converts it into a plain dict:
#          {"text": "Why don't cats play poker? - Too many cheetahs!"}
#          This is needed because prompt2 expects a dict, not a Pydantic object.
#
# Step 4 - prompt2 receives that dict and builds the text:
#          "Explain the following joke - Why don't cats play poker? - Too many cheetahs!"
#
# Step 5 - structured_model2 sends that text to Mistral-7B and forces the output
#          to match the Explanation schema, returning:
#          Explanation(explanation="...", funny_because="...")
#
# Final result is a structured Explanation object with two fields:
#   result.explanation  -> the explanation of the joke
#   result.funny_because -> why the joke is funny

