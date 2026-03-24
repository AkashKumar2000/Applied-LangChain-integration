from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Tesla reported record-breaking quarterly earnings this week, with revenue surging 25% year-over-year to $28 billion, driven by strong demand for its Model Y and the newly launched Cybertruck. The stock jumped 12% in after-hours trading, reaching an all-time high of $320 per share.

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

print(result)