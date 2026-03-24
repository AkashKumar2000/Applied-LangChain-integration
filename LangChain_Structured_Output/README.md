# LangChain Structured Output

This module demonstrates three different approaches to extract **structured output** from LLMs using LangChain's `with_structured_output()` method with OpenAI.

All examples use the same financial news review (Tesla earnings report) as input and extract the same fields from it.

---

## Files

### 1. `structured_output_typeDict.py` — TypedDict Approach
Uses Python's `TypedDict` with `Annotated` to define the schema. The annotations act as instructions to the model for each field.

```python
class ProductReview(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes..."]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment..."]
    pros: Annotated[Optional[list[str]], "Write down all the pros..."]
    cons: Annotated[Optional[list[str]], "Write down all the cons..."]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
```

---

### 2. `structured_output_pydantic.py` — Pydantic BaseModel Approach
Uses Pydantic's `BaseModel` with `Field` for schema definition. Provides built-in validation and is the most production-friendly approach.

```python
class Review(BaseModel):
    key_themes: list[str] = Field(description="...")
    summary: str = Field(description="...")
    sentiment: Literal["pos", "neg"] = Field(description="...")
    pros: Optional[list[str]] = Field(default=None, description="...")
    cons: Optional[list[str]] = Field(default=None, description="...")
    name: Optional[str] = Field(default=None, description="...")
```

---

### 3. `with_structured_output_json.py` — JSON Schema Approach
Uses a raw JSON Schema dictionary to define the output structure. Useful when you need full control or are working with non-Python systems.

```python
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": { ... },
  "required": ["key_themes", "summary", "sentiment"]
}
```

---

## Schema Fields

| Field        | Type                  | Required | Description                          |
|--------------|-----------------------|----------|--------------------------------------|
| `key_themes` | `list[str]`           | Yes      | Key themes discussed in the review   |
| `summary`    | `str`                 | Yes      | Brief summary of the review          |
| `sentiment`  | `"pos"` or `"neg"`    | Yes      | Overall sentiment of the review      |
| `pros`       | `list[str]` or `None` | No       | List of positive points              |
| `cons`       | `list[str]` or `None` | No       | List of negative points              |
| `name`       | `str` or `None`       | No       | Name of the reviewer                 |

---

## Setup

1. Install dependencies:
```bash
pip install langchain langchain-openai python-dotenv pydantic
```

2. Create a `.env` file in the root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run any file:
```bash
python structured_output_typeDict.py
python structured_output_pydantic.py
python with_structured_output_json.py
```

---

## Comparison

| Approach    | Validation | Ease of Use | Flexibility |
|-------------|------------|-------------|-------------|
| TypedDict   | No         | Simple      | Medium      |
| Pydantic    | Yes        | Moderate    | High        |
| JSON Schema | No         | Verbose     | Highest     |

> **Note:** `with_structured_output()` is not supported by HuggingFace models. Use `ChatOpenAI` or other compatible providers.
