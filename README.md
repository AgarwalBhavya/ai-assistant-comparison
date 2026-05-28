# ⚖️ Dual AI Assistant Arena & Evaluation Framework

An enterprise-ready comparison, evaluation, and observability platform that benchmarks an **Open-Source LLM Assistant (Qwen 2.5-0.5B-Instruct)** against a **Frontier LLM Assistant (Google Gemini 1.5 Flash)** side-by-side.

Both assistants support:
- **Multi-turn conversations** with short-term sliding memory and summarization.
- **Dynamic Tool Execution** (AST-parsed calculator, mock web search, host system diagnostic telemetry).
- **Dual-Layer Safety Guardrails** (Input PII scrubbers, prompt injection block, output toxicity filtration, and policy compliance).
- **SQLite Observability Database** (Logging every prompt, token, latency, tool call, and safety flag).
- **High-Fidelity Offline Simulator Fallbacks** (Allowing 100% operation if API keys are missing).

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python 3.13+** installed on your system.
- Standard internet access (for HF and Gemini API queries).

### 2. Project Directory Setup
We recommend setting the project directory as your active workspace:
```bash
C:\Users\agarw\.gemini\antigravity\scratch\ai-assistant-comparison
```

### 3. Install Dependencies
Navigate to the project root and run:
```bash
pip install -r requirements.txt
```

### 4. Configuration (.env)
Copy the `.env.example` file to `.env`:
```bash
copy .env .env
```
Open `.env` and configure your API keys:
- **`GEMINI_API_KEY`**: Obtain from [Google AI Studio](https://aistudio.google.com/). Required for live Frontier Assistant queries.
- **`HF_API_TOKEN`**: Obtain from [Hugging Face Settings](https://huggingface.co/settings/tokens). Recommended for fast serverless OSS Assistant queries.

*Note: If no API keys are provided, the assistants run in a high-fidelity simulator fallback mode, allowing you to test all features (chats, guardrails, tools, evaluations) seamlessly without credentials.*

### 5. Start the Web Dashboard
Launch the stunning interactive Streamlit dashboard:
```bash
streamlit run app.py
```
This opens the Web UI in your default browser (usually at `http://localhost:8501`).

### 6. Run the CLI Evaluation Suite & PDF Generator
To run the automated benchmark dataset against both assistants and generate a professional, 1-page PDF report:
```bash
python generate_report.py
```
This creates:
- **`eval_chart.png`**: High-quality performance comparison bar chart.
- **`eval_report.pdf`**: Publication-ready evaluation report complete with core statistics matrices, infographic charts, and strategic recommendation blocks.

---

## 🏛️ System Architecture Decisions

```mermaid
graph TD
    User([User Prompt]) --> IG[Input Guardrails]
    IG --> |PII Scrub / Injection Shield| Mem[Conversational Memory]
    Mem --> |Sliding Window Context| Base[Assistant Base Coordinator]
    Base --> |Orchestrate Turn| Model{Model Select}
    
    Model --> |HF API / Local / Simulator| OSS[Qwen 2.5 OSS]
    Model --> |Gemini SDK / Simulator| Front[Gemini 1.5 Frontier]
    
    OSS --> Out[Output Synthesis & Tool Parse]
    Front --> Out
    
    Out --> |[TOOL_USE]| Tool[Tool Executor: Math, Search, Stats]
    Tool --> |Append Results| Base
    
    Out --> OG[Output Guardrails]
    OG --> |Toxicity / Refusal Policies| Log[SQLite Observability Logs]
    Log --> Final[Final Response]
```

### Key Architectural Choices:
1. **Hybrid Execution Engine (Triple-Fault Tolerance)**:
   The OSS model leverages the serverless Hugging Face Inference API for speed. If keys are missing or rate limits occur, it dynamically pivots to a smart reactive simulator mode to preserve UI uptime.
2. **Deterministic Safety Guardrails**:
   Unlike LLM-based guardrails which are slow and expensive, we implement an optimized, deterministic guardrail layer (regex PII scrubbing, keyword injection shield) in `guardrails.py` running locally in $<1\text{ms}$.
3. **Double-Buffer Memory Retention**:
   In `memory.py`, we implement a sliding window keeping the last $N$ turns. If the conversation grows longer, the oldest turns are programmatically summarized and prepended to the system prompt to maintain long-term memory without filling prompt tokens.
4. **Local Tool Orchestration Loop**:
   A regex tool-parsing engine in `tools.py` intercepts model instructions matching `[TOOL_USE: tool_name("args")]`, executes them locally (e.g. evaluating arithmetic safely via AST tree nodes rather than dangerous raw `eval`), and feeds the result back to the model for final synthesis.

---

## ⚖️ Tradeoffs & Design Evaluations

| Architectural Aspect | Chosen Approach | Tradeoff Made | Mitigating Factor |
| :--- | :--- | :--- | :--- |
| **OSS Model Selection** | `Qwen 2.5-0.5B-Instruct` | 0.5B parameters have lower cognitive capacity than 7B+ models. | Runs extremely fast on free CPU/serverless tiers, and can be easily hosted. |
| **Security Guardrails** | Deterministic Regex & Keyword | May fail to catch extremely complex, semantic jailbreaks. | Lightweight, runs in microseconds, and costs $0.00 in API tokens. |
| **Tool Execution** | Regex Match Interceptor | Less flexible than native JSON schema function calling. | Uniformly supported across both lightweight OSS models and frontier models. |
| **Evaluation Method** | Heuristic Programmatic Scorer | Lacks semantic flexibility of an LLM-as-a-judge scorer. | 100% deterministic, repeatable, runs completely offline, and costs $0.00. |

---

## 🔮 What We Would Improve With More Time

1. **Production LLM-as-a-Judge**: Integrate a secondary Gemini-1.5-Pro evaluator instance to run advanced semantic evaluations, scoring tone, helpfulness, and conversational alignment.
2. **Vector DB Semantic Memory**: Swap sliding-window memory for local SQLite-based vector retrieval (using `chromadb` or sentence-transformers) to pull highly relevant historical context dynamically.
3. **Guardrail Enhancements**: Implement **Llama Guard 3** or **NeMo Guardrails** as a containerized security sidecar for enterprise-grade adversarial protection.
4. **Online Observability Integrations**: Push observability logs from `database.db` to standard cloud tracing platforms like **Arize Phoenix**, **LangSmith**, or **Literal AI** to enable remote monitoring.
