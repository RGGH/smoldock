# SmolAgents + Qdrant Financial Advice* RAG Agent

(*example, not to be used as actual financial advice)

This project is a small retrieval-augmented agent built with:

- 🧠 `smolagents` for tool-using LLM orchestration
- 🔎 `Qdrant` for vector search over structured knowledge fragments
- 🧬 `fastembed` for embeddings
- 🦙 a local LLM served via Ollama (`qwen3-coder` in this setup)

The system is designed to produce **structured financial hardship advice** by retrieving small, focused knowledge fragments and then synthesising them into an actionable plan.

<p align="center">
  <a href="https://youtu.be/7kmfSxv1Sbk">
    <img src="https://img.youtube.com/vi/7kmfSxv1Sbk/maxresdefault.jpg" width="600" alt="Demo video thumbnail — click to watch on YouTube">
  </a>
  <br>
  <em>Click the image above to watch the demo ▶️</em>
</p>

---

## 🔧 What It Does

Instead of storing full answers, the system stores **fine-grained financial facts**, such as:

- Definitions (e.g. what mortgage arrears are)
- Regulations (e.g. FCA rules)
- Remedies (e.g. payment plans, refinancing options)
- Risks (e.g. credit impact)
- Support schemes (e.g. UK SMI)

When given a real-world scenario, the agent:

1. Queries a vector database (Qdrant)
2. Retrieves multiple relevant fragments
3. Combines them using an LLM
4. Produces a **personalised 4–6 step action plan**

---

## 🧠 Architecture

```

User Scenario
↓
CodeAgent (smolagents)
↓
QdrantQueryTool (semantic retrieval)
↓
Qdrant Vector DB (embedded financial knowledge)
↓
LLM (Ollama / Qwen3 Coder)
↓
Synthesised Action Plan

````

---

## 📦 Knowledge Base

The embedded dataset (`FINANCIAL_KNOWLEDGE`) includes:

- Mortgage arrears rules and forbearance options
- Credit card debt spirals
- Payday loan regulations
- Council tax arrears processes
- Overdraft dependency risks
- Student loan repayment structures
- Business cash flow issues
- Medical debt handling
- Retirement shortfalls
- Identity theft recovery steps

Each entry is intentionally **atomic** (one fact per chunk) to improve retrieval quality and compositional reasoning.

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install fastembed qdrant-client smolagents litellm
````

You also need a running Qdrant instance:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

### 2. Run Ollama model

Make sure Ollama is running locally:

```bash
ollama run qwen3-coder:30b
```

Update the model config if needed:

```python
model_id="ollama/qwen3-coder:30b"
api_base="http://localhost:11434"
```

---

### 3. Run the agent

```bash
python main.py
```

---

## 🧩 How the Tool Works

The core tool is `QdrantQueryTool`, which:

* Embeds a query using `jinaai/jina-embeddings-v2-base-en`
* Searches Qdrant for the most relevant knowledge fragments
* Returns top-k results with scores and metadata

Each result contains:

* Topic (e.g. Mortgage Arrears)
* Type (definition, remedy, regulation, etc.)
* Raw fact text
* Similarity score

---

## 🤖 Agent Behaviour

The agent is explicitly instructed to:

* Run **multiple retrieval queries**
* Cover different angles (risk, regulation, remedy, etc.)
* Avoid relying on a single chunk of knowledge
* Produce a structured, contextual action plan
* Only use retrieved facts (no hallucinated policy or advice)

---

## 🧪 Example Scenario

The default scenario:

> A UK homeowner has missed two mortgage payments, is self-employed, has no savings buffer, and has received an arrears warning letter.

The agent is expected to produce:

* Prioritised steps (contact lender, check forbearance options, etc.)
* References to UK-specific regulation (MCOB rules)
* Debt support pathways (e.g. StepChange, SMI scheme)
* Risk explanation (credit impact, repossession process)

---

## ⚙️ Key Design Choices

### 1. Fine-grained knowledge chunks

Each fact is atomic to allow recombination during reasoning.

### 2. Multi-query retrieval

The agent is encouraged to run multiple semantic searches instead of relying on a single embedding result.

### 3. Tool-based architecture

The LLM does not directly “know” finance — it must query tools.

### 4. Local-first stack

* Qdrant runs locally
* Ollama runs locally
* No external API dependency required

---

## 📁 Project Structure

```
.
├── main.py              # Agent + tool + knowledge base
├── pyproject.toml      # Python project config
├── qdrant_storage/     # Local vector DB state (ignored in git)
├── output.md           # Generated outputs (optional)
```

---

## 🧠 Potential Improvements

* Add document ingestion pipeline (PDFs, scraping, etc.)
* Introduce query planner agent (decides which angles to search)
* Add reranking stage for better retrieval quality
* Persist Qdrant collection instead of re-seeding each run
* Add structured output schema for action plans

---

## ⚠️ Disclaimer

This system is for experimental and educational purposes. It is not a substitute for professional financial advice.

---

## 📜 License

MIT 

