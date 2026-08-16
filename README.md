# Nissan Moco 2012 RAG Manual Assistant

A conversational RAG (Retrieval-Augmented Generation) system that reads the **Japanese owner's manual** for the Nissan Moco 2012 and answers questions entirely in **English** — with accurate page citations.

Built with **LangGraph**, **LangChain**, **Groq**, and **Cohere**.

---

## Demo

![Chat Interface](docs/screenshot-chat.png)

> **User:** What kind of radiator coolant does the car use?  
> **Assistant:** The car uses Suzuki Super Long Life Coolant (blue) as the radiator coolant, as stated on [Page 281]. The specified quantity is 3.3 L for non-turbo cars without idling stop and 3.2 L for non-turbo cars with idling stop.

> **User:** What quantity is it used in?  
> **Assistant:** The specified quantity of Suzuki Super Long Life Coolant used in non-turbo cars with idling stop is 3.2 L, as stated on [Page 281].

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│   Rewrite    │────▶│  Retrieve   │────▶│  Generate   │
│   Query     │     │   (LLM)      │     │  (Rerank)   │     │   (LLM)     │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    Standalone query      Top-8 relevant
                    (context baked in)    manual chunks
```

### Key Design Decisions

| Component | Implementation | Rationale |
|-----------|---------------|-----------|
| **Orchestration** | `StateGraph` (LangGraph) | Deterministic pipeline (rewrite → retrieve → generate) instead of flaky ReAct agent loops |
| **Query Rewriting** | `llama-3.1-8b-instant` | Follow-up questions like *"What quantity?"* are rewritten into standalone queries with full context |
| **Cross-lingual Retrieval** | English → Japanese translation | Japanese manual indexed in Japanese; queries translated for better semantic search |
| **Reranking** | Cohere `rerank-multilingual-v3.0` | Retrieves 100 chunks, reranks to top 8 for quality |
| **Context Capping** | 800 chars per chunk | Fits within Groq's free-tier token limits while preserving tables |
| **Stateless Generation** | No conversation history in LLM prompt | Prevents cross-turn hallucination and topic bleed |
| **Evaluation** | LLM-as-judge + deterministic citation check | Automated faithfulness, correctness, and citation scoring |

---

## Features

- **Conversational follow-ups** — understands context across turns (*"What quantity is it used in?"* after asking about coolant)
- **Accurate citations** — cites exact page numbers from the original PDF (e.g., [Page 281])
- **No hallucination** — answers strictly from retrieved manual sections; falls back to *"I couldn't find this..."* when appropriate
- **No Japanese in output** — all text translated to English, including product names and technical terms
- **Web UI + CLI** — chat via browser or terminal
- **Automated evaluation** — run `npm run eval-llm` to score the system against 17 test cases

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq (`llama-3.3-70b-versatile` / `llama-3.1-70b-versatile`) |
| Embeddings | Cohere `embed-multilingual-v3.0` |
| Reranking | Cohere `rerank-multilingual-v3.0` |
| Vector Store | LangChain `MemoryVectorStore` |
| PDF Parsing | LangChain `PDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (2500 chars, 500 overlap) |
| Orchestration | LangGraph `StateGraph` |
| Backend | Express.js |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
.
├── src/
│   ├── agent.ts              # LangGraph StateGraph (rewrite → retrieve → generate)
│   ├── retriever.ts          # Query translation, reranking, chunk formatting
│   ├── ingest.ts             # PDF loading, chunking, embedding, vector store creation
│   ├── index.ts              # CLI chat interface
│   ├── server.ts             # Express web server with session management
│   └── llm_evaluator_manual_groundtruth.ts  # Automated evaluation suite
├── data/
│   └── moco-2012.pdf         # Nissan Moco 2012 owner's manual (Japanese)
├── public/
│   └── index.html            # Web UI
├── .env                      # API keys (see .env.example)
├── package.json
└── tsconfig.json
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd Mocco-2012-RAG-Manual
npm install
```

### 2. Add API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
PORT=3000
```

- Get a **Groq** key at [console.groq.com](https://console.groq.com)
- Get a **Cohere** key at [cohere.com](https://cohere.com)

### 3. Add the manual

Place `moco-2012.pdf` in the `data/` directory.

---

## Usage

### Web UI

```bash
npm run dev
# or
npm start
```

Visit `http://localhost:3000`

### CLI

```bash
npx ts-node src/index.ts
```

### Evaluation

Run the full test suite (17 test cases, 3 runs each):

```bash
npm run eval-llm
```

Run a single test case:

```bash
npm run eval-llm -- --test=TC002 --runs=1
```

---

## Evaluation Methodology

The evaluator scores each answer on three metrics (0–10):

1. **Faithfulness** — every claim must be directly supported by retrieved chunks
2. **Correctness** — facts must match the manual
3. **Citation Accuracy** — every cited page must appear in the retrieved context

Two independent checks:
- **Deterministic regex checker** — verifies page numbers exist in retrieved text
- **LLM judge** (`llama-3.3-70b-versatile`) — evaluates semantic correctness

Final citation score = `min(deterministic_score, llm_score)` for strictness.

### Sample Results

| Test Case | Question | Faithfulness | Correctness | Citation |
|-----------|----------|-------------|-------------|----------|
| TC001 | What is the fuel tank capacity? | 10.0 | 10.0 | 10.0 |
| TC002 | What type of coolant does the car use? | 10.0 | 10.0 | 10.0 |
| TC004 | What is the recommended tire pressure? | 10.0 | 10.0 | 10.0 |

---

## Known Limitations

- **Groq free-tier rate limits**: 100,000 tokens/day. Heavy testing may trigger rate limiting. Switch to `llama-3.1-70b-versatile` in `agent.ts` for lower token consumption, or upgrade to Groq Dev Tier.
- **In-memory vector store**: Data is lost on restart. For production, swap `MemoryVectorStore` for a persistent store (e.g., Pinecone, Weaviate, or PostgreSQL with `pgvector`).
- **Chunk overlap**: Some multi-page tables may be split across chunks. The reranker usually recovers both halves, but edge cases exist.

---

## Why This Architecture?

### From ReAct Agent → Deterministic Graph

Early versions used `createReactAgent` (ReAct loop). The LLM decided whether/how many times to retrieve — leading to:
- Multiple retriever calls per turn (token waste)
- Cross-turn hallucination (stitching unrelated pages)
- Inconsistent answers

The current `StateGraph` pipeline enforces **exactly one** rewrite, **exactly one** retrieval, and **exactly one** generation per turn. No loops, no ambiguity.

### Stateless Generation

Passing full conversation history to the LLM caused topic bleed: the coolant answer would contaminate an uphill driving question. By making generation stateless (only the standalone query + retrieved chunks), each turn is independent and grounded.

---

## License

MIT
