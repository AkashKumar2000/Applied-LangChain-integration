
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = ChatOpenAI()

#Annoted is used to tell me extra information to chat model , what specifically we want for this parameter.
# schema

class ProductReview(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


structured_model = llm.with_structured_output(ProductReview)

extracted_review = structured_model.invoke("""Tesla reported record-breaking quarterly earnings this week, with revenue surging 25% year-over-year to $28 billion, driven by strong demand for its Model Y and the newly launched Cybertruck. The stock jumped 12% in after-hours trading, reaching an all-time high of $320 per share.

However, the company also warned of rising production costs due to supply chain disruptions and increasing raw material prices for lithium batteries. Profit margins slightly declined from 18% to 15%, raising concerns among some analysts about long-term sustainability.

The Federal Reserve’s decision to hold interest rates steady provided additional relief to tech and EV stocks, boosting investor confidence across the board. Meanwhile, Tesla’s expansion into India and Southeast Asia is expected to open massive new revenue streams in 2025.

On the downside, increased competition from BYD and Rivian is putting pressure on Tesla’s market share in the affordable EV segment. Analysts remain divided — some see this as a golden buying opportunity, while others caution that the stock is overvalued at current levels.

Pros:
Record revenue and strong global demand
New market expansion into Asia
Positive investor sentiment post Fed decision

Cons:
Declining profit margins
Rising production costs
Increasing competition from BYD and Rivian

""")

print(extracted_review['name'])

# Here with_structured_output does not support Hugging face model so we have to switch to chatOpenAI or any other

"""from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
import os 

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token= hf_token
)

#Schema 

class Review(TypedDict):
    summary:str
    sentiment:str

structured_model=llm.with_structured_output(Review)

result=structured_model.invoke("
The hardware is great , but the software feels bloated. There are too many pre-installed apss that 
I can't remove . Also, the UI looks outdated compared to other brands . Hoping for a software update to fix this
")

print(result)
print(result['summary'])
print(result['sentiment'])

"""