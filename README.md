# 📘 LangSmith – Observability for LLM Applications (Part 1)

---

## 1️⃣ Introduction: What is this about?

In this part, we learn about **LangSmith**, an important tool in **Generative AI (GenAI)**.

So far, our GenAI learning path looks like this:

* Started with **LangChain**
* Moved to **LangGraph** (agentic workflows)

While building LangGraph-based applications, we hit a very important requirement:

👉 **Observability for LLM applications**

That is why we take a short detour and learn **LangSmith**.

---

## 2️⃣ Why do we need LangSmith?

LangSmith is:

* A powerful **observability platform**
* Used to **debug**, **monitor**, and **evaluate** LLM applications
* Works seamlessly with **LangChain** and **LangGraph**

Before writing code, it is important to understand **why** such a tool is needed using real-world problems.

---

## 3️⃣ Key Concept: Observability (VERY IMPORTANT)

### 🔹 What is Observability?

**Simple definition:**

Observability means understanding **what is happening inside a system** by analyzing:

* Logs
* Metrics
* Traces

In simple words:

👉 Observability helps you understand **WHY something is happening**, even when the problem is unexpected.

---

## 4️⃣ Why Observability is hard in LLM systems

LLM-based systems are:

❌ **Non-deterministic**
Same input does not always give the same output

❌ **Complex**
Multiple components like:

* LLM calls
* Agents
* Tools
* RAG pipelines
* Loops

❌ **Black boxes**

* No proper error stack trace
* No traditional debugging

When things go wrong, we see:

* High latency
* High cost
* Hallucinations

👉 Debugging becomes extremely difficult.

---

## 5️⃣ Use Case 1: Latency Problem in LLM Workflow

### 🧠 Scenario

You build a **Job Application Assistant** for students:

Steps:

1. Read Job Description (JD)
2. Fetch resume from Google Drive
3. Match skills
4. Generate cover letter
5. Proofread

Normal behavior:

* ⏱️ Takes ~2 minutes

Sudden issue:

* ⏱️ Takes 7–10 minutes
* Users complain
* Revenue loss

### ❌ Problem

You only know:

* User input
* Final output
* Total time

You **do NOT know**:

* Which step is slow
* JD reading?
* Resume fetching?
* Matching?
* Generation?

👉 No internal visibility
👉 Debugging becomes guesswork

### ✅ How LangSmith helps

LangSmith shows:

* Step-by-step execution
* Time taken by each component
* Exact bottleneck

---

## 6️⃣ Use Case 2: Cost Explosion in Agent-Based System

### 🧠 Scenario

You build a **Research Assistant Agent**:

Steps:

* Fetch papers (Google Scholar / arXiv)
* Read papers
* Extract key points
* Summarize report
* Allow chat on report

Cost behavior:

* Earlier: ₹0.50 per report
* Suddenly: ₹2 per report

### ❌ Problem

* Some reports are cheap
* Some are very expensive
* No errors
* No crashes

Agent mistake example:

> "Keep improving the report until it becomes perfect"

Agent behavior:

* Loops internally
* Repeats:

  * Fetch → Read → Summarize → Evaluate → Repeat

Result:

* 🔥 Token usage increases
* 💰 Cost explodes
* 🧩 Hard to debug

### ✅ How LangSmith helps

LangSmith shows:

* How many times the agent looped
* Which steps repeated
* Token usage per step
* Cost per execution

---

## 7️⃣ Use Case 3: Hallucinations in RAG System

### 🧠 Scenario

You build a **RAG-based chatbot** for a company (e.g., TCS):

Knowledge base:

* HR policies
* Leave policy
* Notice period
* Insurance

### ❌ Problem: Hallucination

Example wrong answer:

> "You can take leave anytime and go to Goa"

This causes:

* Misinformation
* Serious company issues

### ❓ Why hallucinations happen in RAG

#### 🔹 1. Retriever Issue

* Wrong documents fetched
* Irrelevant context

Example:

* Question: Notice period
* Retrieved doc: Company history

#### 🔹 2. Generator (LLM) Issue

* Weak prompt
* Low-quality model
* Prompt does not enforce *"answer only from context"*

### ❌ Debugging problem

You cannot see:

* Which documents were retrieved
* Final prompt sent to LLM
* Whether retriever or generator failed

### ✅ How LangSmith helps

LangSmith shows:

* Retrieved documents
* Prompt sent to LLM
* LLM output
* Full step-by-step trace

---

## 8️⃣ Common Problem in All Scenarios

All systems suffer from:

❌ No internal visibility
❌ Black-box behavior
❌ Hard debugging

👉 We need a tool that converts **Black Box → White Box**

---

## 9️⃣ What is LangSmith? (Formal Definition)

**LangSmith** is:

> A unified observability and evaluation platform that helps teams debug, test, and monitor LLM application performance.

### In simple words:

LangSmith records:

* Inputs
* Outputs
* Intermediate steps
* Latency
* Token usage
* Cost
* Errors
* Metadata

---

## 🔟 What does LangSmith Trace?

LangSmith tracks:

* ✅ User input & final output
* ✅ Intermediate steps (chains, agents, RAG)
* ✅ Latency (component-wise)
* ✅ Token usage (input + output)
* ✅ Cost estimation
* ✅ Errors
* ✅ Tags
* ✅ Metadata
* ✅ Optional user feedback

---

## 1️⃣1️⃣ Core Concepts in LangSmith

### 🔹 1. Project

* Represents the entire LLM application
* Example: Chatbot, RAG app, Agent

### 🔹 2. Trace

* One full execution of the application
* Example: One user query → one response

### 🔹 3. Run

* Execution of a single component
* Example:

  * Prompt
  * LLM call
  * Output parser

### 📌 Hierarchy

```
Project
 └── Trace (one execution)
      └── Runs (each component)
```

---

## 1️⃣2️⃣ Setting Up LangSmith (Practical)

### 🔹 Step 1: Clone Repository

```bash
git clone <repo-url>
cd langsmith-masterclass
```

### 🔹 Step 2: Create Virtual Environment

```bash
python -m venv myenv
```

Activate:

```bash
myenv\Scripts\activate   # Windows
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 4: Create LangSmith Account

* Visit LangSmith website
* Sign up / Login
* Generate **Personal Access Token**

### 🔹 Step 5: Create `.env` file

```env
OPENAI_API_KEY=your_openai_key

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=langsmith-demo
```

---

## 1️⃣3️⃣ First LangSmith Trace (Simple LLM App)

### 🔹 LangChain Code

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Answer: {question}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"question": "What is the capital of Peru?"}))
```

### 🔹 Important Point

❗ No LangSmith-specific code is written.

LangSmith automatically traces because:

* Environment variables are enabled

### 🔹 What you see in LangSmith UI

* Project: `langsmith-demo`
* One Trace per execution
* Runs:

  * PromptTemplate
  * ChatOpenAI
  * OutputParser

You can inspect:

* Inputs / Outputs
* Latency
* Tokens
* Cost

---

## 1️⃣4️⃣ Setting Project Name from Code

```python
import os
os.environ["LANGCHAIN_PROJECT"] = "sequential-llm-app"
```

This overrides the project name from `.env`.

---

## 1️⃣5️⃣ Adding Tags & Metadata

```python
config = {
    "tags": ["llm-app", "report-generation", "summarization"],
    "metadata": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "parser": "StrOutputParser"
    },
    "run_name": "Sequential Chain"
}

chain.invoke({"topic": "Unemployment in India"}, config=config)
```

---

## 1️⃣6️⃣ What We Learned in Part 1

✅ Why observability is critical for LLM apps
✅ Latency, cost & hallucination problems
✅ What LangSmith is and why it exists
✅ Core concepts: Project, Trace, Run
✅ Automatic tracing in LangChain apps
✅ How to add:

* Project names
* Tags
* Metadata

---

📌 **Next Part:** LangSmith evaluations, debugging strategies, and real-world workflows
