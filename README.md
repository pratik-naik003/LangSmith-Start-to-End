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

# 📘 LangSmith – Tracing RAG, Agents & LangGraph

*(Part 2 – Simple English Notes with Code)*

---

## 1️⃣ Why LangSmith + RAG is a Very Good Idea

### 🔹 What is a RAG application?

**RAG = Retrieval Augmented Generation**

In a RAG app:

1. User asks a question
2. Retriever fetches relevant documents
3. LLM receives:

   * The question
   * Retrieved context
4. LLM combines both and generates the final answer

👉 Used for:

* PDFs
* Company documents
* Personal data
* Knowledge bases

---

## 2️⃣ The Real Problem with RAG in Production

Even though RAG sounds simple, many production systems fail.

### ❌ Two common error types

#### 🔴 Error Type 1: Retriever Error

* Retriever fetches wrong / irrelevant chunks
* LLM receives bad context
* Final answer becomes incorrect

#### 🔴 Error Type 2: Generator (LLM) Error

* Retriever fetches correct chunks
* LLM hallucinates or ignores context
* Final answer is still wrong

### ❌ The BIG Production Problem

You only see:

* User question
* Final answer

You **cannot see**:

* What documents were retrieved
* What exact prompt was sent to the LLM

👉 No intermediate visibility

---

## 3️⃣ How LangSmith Solves This Problem

LangSmith traces **every intermediate step**:

It records:

* User question
* Retrieved documents
* Final prompt (question + context)
* LLM response

👉 Now you can clearly identify:

* Retriever failure ❌
* Generator failure ❌

---

## 4️⃣ Simple RAG App Used in Demo

### 📄 Data

* PDF: *Introduction to Statistical Learning*
* Stored locally inside project folder

### 🧠 Example Queries

* "Who is the author of this book?"
* "Summarize chapter 6"

---

## 5️⃣ RAG Application Flow (Very Important)

### 🔁 Step-by-step Flow

1. Load PDF
2. Split PDF into chunks
3. Generate embeddings
4. Create retriever
5. Pass:

   * Question
   * Retrieved context
6. LLM generates final answer

---

## 6️⃣ Core RAG Code Structure (Simplified)

### 🔹 Load PDF

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("book.pdf")
documents = loader.load()
```

### 🔹 Split Documents

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)
```

### 🔹 Create Embeddings & Retriever

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()
```

### 🔹 Prompt Template (VERY IMPORTANT)

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    Answer ONLY from the provided context.
    If answer not found, say "I don't know".

    Question: {question}
    Context: {context}
    """
)
```

---

## 7️⃣ RAG Chain Structure (Conceptual)

**Parallel Chain**

* Path 1 → Question (unchanged)
* Path 2 → Question → Retriever → Context

Outputs:

* Question
* Context

Then:

Prompt → LLM → Output Parser

---

## 8️⃣ Setting LangSmith Project Name

```python
import os
os.environ["LANGCHAIN_PROJECT"] = "rag-chatbot"
```

---

## 9️⃣ What LangSmith Shows in UI

LangSmith beautifully visualizes:

* Entire RAG chain
* RunnableParallel
* Retriever calls
* Prompt template
* LLM calls
* Token usage
* Latency per step
* Cost

---

## 🔟 Problem #1: Partial Tracing ❌

### ❌ Issue

LangSmith was tracing only:

* Chain execution

It was **NOT tracing**:

* PDF loading
* Chunking
* Embeddings

👉 Because LangSmith auto-traces only **LangChain Runnables**

---

## 1️⃣1️⃣ Solution: `@traceable` Decorator

### 🔹 Import

```python
from langsmith import traceable
```

### 🔹 Convert Steps into Traceable Functions

```python
@traceable(name="Load PDF")
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name="Split Documents")
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

@traceable(name="Build Vector Store")
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()
```

### 🔹 Pipeline Function

```python
@traceable(name="Setup Pipeline")
def setup_pipeline(pdf_path):
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    retriever = build_vectorstore(chunks)
    return retriever
```

---

## 1️⃣2️⃣ Result in LangSmith UI

Now LangSmith shows:

* Setup Pipeline (trace)
* Load PDF
* Split Documents
* Build Vector Store
* RAG Query (trace)

Each step displays:

* Inputs
* Outputs
* Time taken
* Metadata

---

## 1️⃣3️⃣ Adding Tags & Metadata (Advanced)

```python
@traceable(
    name="Build Vector Store",
    tags=["embedding", "vectorstore"],
    metadata={
        "embedding_model": "text-embedding-3-small",
        "dimensions": 1536
    }
)
def build_vectorstore(chunks):
    ...
```

👉 Helps in:

* Searching traces
* Debugging large systems
* Monitoring specific components

---

## 1️⃣4️⃣ Problem #2: High Latency ❌

### ❌ Issue

Every query:

* Reloads PDF
* Re-chunks
* Re-embeds

➡️ Extremely slow (200+ seconds)

---

## 1️⃣5️⃣ Solution: Persistent Vector Store (FAISS)

### 🔹 Concept

* First run → Build index
* Save index to disk
* Next runs → Load index

### 🔹 Logic (Conceptual)

```python
if index_exists():
    load_index()
else:
    build_index()
    save_index()
```

### 🔹 Performance Benefit

| Scenario  | Time     |
| --------- | -------- |
| First run | ~30 sec  |
| Next runs | ~1–4 sec |

---

## 1️⃣6️⃣ When Is Index Rebuilt?

Index rebuild happens when:

* PDF content changes
* PDF metadata changes
* Chunk size / overlap changes
* Embedding model changes

---

## 1️⃣7️⃣ Key Production Lesson (VERY IMPORTANT)

👉 **Never rebuild embeddings on every query**

Always:

* Pre-build vector index
* Reuse embeddings

---

# 🧠 Agent Tracing with LangSmith

## 1️⃣ Why Agent Tracing Matters

Agents are:

* Autonomous
* Multi-step
* Tool-using
* Non-deterministic

👉 Debugging agents **without tracing is impossible**

---

## 2️⃣ Agent Example Used

Tools:

* DuckDuckGo Search
* Weather API

Agent loop:

**Thought → Action → Observation → Repeat**

---

## 3️⃣ What LangSmith Shows for Agents

LangSmith traces:

* Scratchpad
* Prompt
* Tool calls
* Tool outputs
* Updated scratchpad
* Final answer

---

## 4️⃣ Example Agent Flow

**Query:**

> What is the current temperature of Gurgaon?

**Steps:**

1. Thought: I should use weather tool
2. Action: Call weather API
3. Observation: Weather data
4. Thought: I now know the answer
5. Final answer

👉 Every step is visible in LangSmith

---

## 5️⃣ Multi-tool Agent Example

**Query:**

> Find birth place of Kalpana Chawla and give its temperature

Agent uses:

* Search tool → Birthplace
* Weather tool → Temperature

LangSmith shows:

* Tool selection
* Inputs & outputs
* Reasoning chain

---

## 6️⃣ Why This Is HUGE

You can:

* Debug hallucinations
* Track cost
* Track tokens
* Understand agent reasoning
* Improve prompts

---

# 🔗 LangGraph + LangSmith Integration

## 1️⃣ LangGraph Basics (Quick Recap)

* LLM apps as workflows
* Nodes = tasks
* Edges = execution flow

Supports:

* Parallel execution
* Conditional branches
* Loops

---

## 2️⃣ LangSmith Integration Concept

### 🔹 Two Golden Rules

1️⃣ Entire graph execution = **One Trace**
2️⃣ Each node execution = **One Run**

---

## 3️⃣ Example: Essay Evaluation Graph

**Input:**

* Essay text

**Nodes:**

* Language evaluation
* Analysis evaluation
* Clarity evaluation

**Final Node:**

* Overall feedback
* Average score

---

## 4️⃣ What LangSmith Shows

* Parallel node execution
* Node-wise latency
* Node-wise cost
* Inputs & outputs
* Structured outputs

---

## 5️⃣ Structured LLM Outputs (Important)

```python
llm = ChatOpenAI().with_structured_output(EvaluationSchema)
```

Ensures:

* Fixed schema
* Reliable outputs
* Easy debugging

---

## 6️⃣ Why LangSmith is PERFECT for LangGraph

Because:

* Graphs are complex
* Branching is hard to debug
* LangSmith visualizes everything

---

# 🌟 Other Important Features of LangSmith

## 1️⃣ Monitoring & Alerting

### 🔹 Monitoring

Analyze multiple traces together to track:

* Latency
* Cost
* Token usage
* Error rate

### 🔹 Alerts

Examples:

* Latency > 5s → alert team
* Cost spike → notify team

👉 Prevents silent production failures

---

## 2️⃣ Evaluation (LLMOps)

Used to:

* Compare model versions
* Compare prompts
* Prevent regressions

Supports:

* LLM-as-a-judge
* Faithfulness
* Relevance
* Custom Python evaluators

---

## 3️⃣ Prompt Experimentation (A/B Testing)

* Compare Prompt A vs Prompt B
* Same dataset
* Same metrics
* Stored history

👉 Scientific prompt engineering

---

## 4️⃣ Dataset Creation & Annotation

* Create datasets from traces
* Add annotations
* Reuse datasets across projects

---

## 5️⃣ User Feedback Integration

* Thumbs up / down
* Structured feedback
* Linked to traces

Helps:

* Improve real-world quality
* Understand user sentiment

---

## 6️⃣ Collaboration

* Share trace links
* Team debugging
* Shared dashboards
* Prompt versioning

---

# 🔚 Final Summary

LangSmith is **NOT just observability**.

It is a complete **LLM Ops platform**:

✅ Observability
✅ Debugging
✅ Monitoring & Alerts
✅ Evaluation
✅ Prompt experimentation
✅ Dataset creation
✅ User feedback
✅ Collaboration

