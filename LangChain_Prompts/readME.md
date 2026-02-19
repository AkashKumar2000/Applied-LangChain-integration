# 🧠 LangChain Chatbot & Prompt Engineering Fundamentals

This project demonstrates the core foundations of building intelligent LLM-based applications using **LangChain**.

It covers:

- What is a Prompt?
- Static vs Dynamic Prompts
- Chat Message Structure
- Conversation History (Memory Simulation)
- Building a Simple Chatbot
- HuggingFace Model Integration

---

# 🚀 Project Overview

Large Language Models (LLMs) do not simply take plain text input.  
They operate using structured chat messages and carefully designed prompts.

This project demonstrates how to:

- Control model behavior using system prompts
- Create dynamic prompts with user inputs
- Maintain chat history manually
- Simulate memory in stateless LLMs
- Build a simple terminal-based chatbot using LangChain

---

# 📌 1️⃣ What is a Prompt?

A **prompt** is the instruction or input given to a Large Language Model to guide its output.

Think of it as:

> Instruction + Context + Query

Example:

```
You are a poetry expert. Write a haiku about spring.
```

Prompt quality directly affects:

- Accuracy
- Tone
- Structure
- Creativity
- Hallucination rate

Prompt engineering is the foundation of modern AI systems.

---

# 📌 2️⃣ Static vs Dynamic Prompts

## 🔹 Static Prompt

A static prompt is fixed text that never changes.

Example:

```python
prompt = "Explain GPT-3 in simple terms."
```

Use cases:
- Simple tasks
- Testing
- Fixed behavior instructions

---

## 🔹 Dynamic Prompt

A dynamic prompt includes variables that change based on user input.

Example:

```python
template = "Explain {topic} in {style} style."
```

If:

```python
topic = "Transformers"
style = "technical"
```

Final prompt becomes:

```
Explain Transformers in technical style.
```

Dynamic prompts allow:

- Personalization
- User interaction
- Scalable AI applications

In LangChain, this is done using:

- `PromptTemplate`
- `ChatPromptTemplate`

---

# 📌 3️⃣ Chat Message Structure in LangChain

Modern chat models use structured messages instead of plain text.

LangChain provides message types:

| Message Type      | Purpose |
|------------------|----------|
| `SystemMessage`  | Sets behavior/personality |
| `HumanMessage`   | User input |
| `AIMessage`      | Model response |
| `ToolMessage`    | Tool output (advanced usage) |

Example:

```python
from langchain.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a poetry expert."),
    HumanMessage(content="Write a haiku about spring.")
]
```

This mirrors how chat models are trained internally.

---

# 📌 4️⃣ What is Chat History?

LLMs are stateless.

They do NOT remember previous messages automatically.

To simulate memory, we pass the full conversation every time.

Example:

```python
chat_history = [
    SystemMessage(content="You are a poetry expert."),
    HumanMessage(content="Write a haiku."),
    AIMessage(content="Cherry blossoms bloom...")
]
```

Each new message is appended:

```python
chat_history.append(HumanMessage(content="Continue it"))
```

Then the entire history is sent again to the model.

---

# 📌 5️⃣ How Memory Works Internally

Every time we call:

```python
model.invoke(chat_history)
```

LangChain converts messages into:

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

The model generates the next response based on full conversation context.

Memory = Re-sending conversation history.

---

# 📌 6️⃣ Simple Chatbot Architecture

Basic flow:

```
User Input
   ↓
Append to Chat History
   ↓
Send Full History to Model
   ↓
Receive AI Response
   ↓
Append AI Response to History
```

This loop continues until user exits.

---

# 📌 7️⃣ Technologies Used

- Python
- LangChain
- HuggingFace Models
- Prompt Templates
- Structured Chat Messages
- Environment Variables (.env)

---

# 📌 8️⃣ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set HuggingFace Token

Create a `.env` file:

```
HUGGINGFACEHUB_ACCESS_TOKEN=your_token_here
```

### 3️⃣ Run the chatbot

```bash
python chatbot.py
```

Type `exit` to stop the chatbot.

---

# 📌 9️⃣ Future Improvements

- Add streaming responses
- Add memory trimming (token limit handling)
- Convert to Streamlit web app
- Add RAG (Retrieval-Augmented Generation)
- Deploy using Docker

---

# 🏁 Conclusion

This project demonstrates the foundational concepts required to build scalable LLM applications:

- Prompt engineering
- Dynamic input handling
- Structured chat communication
- Manual memory simulation
- Model integration

Understanding these concepts is essential for building advanced systems like:

- Research assistants
- AI tutors
- Finance bots
- RAG systems
- Autonomous agents
