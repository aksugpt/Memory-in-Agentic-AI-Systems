# memory-write-gate

Phase 0/1 starter code: a working agent with long-term memory
(ChromaDB), built as the vulnerable baseline for the memory-injection
attack + write-gate defense project.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste in your real GROQ_API_KEY
# (get one free at https://console.groq.com -- no credit card needed)
```

Uses Groq for both extraction (`openai/gpt-oss-20b`) and chat
(`openai/gpt-oss-120b`) -- free tier gives far more daily requests
than Gemini's free tier, which matters once you're running dozens of
trials for your ASR measurements. Note: these replace Groq's older
`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`, which were
deprecated in June 2026 -- if you see a model-not-found error, check
https://console.groq.com/docs/deprecations for the current
recommended models.

## Files so far

- `memory_store.py` — thin ChromaDB wrapper: `write()`, `retrieve()`,
  `all_memories()`, `reset()`. No security logic lives here.
- `extraction.py` — one LLM call that decides if a piece of text is
  worth remembering. Deliberately naive: no source-awareness yet.
- `agent.py` — the chat loop. `chat()` for normal conversation,
  `process_document()` for the "agent reads a document/email" flow
  that Phase 1's indirect-injection attack will use.

## Try it

```bash
python memory_store.py     # standalone smoke test of the vector store
python extraction.py       # standalone test of the extraction LLM call
python agent.py            # full demo: tell it a fact, start a fresh
                            # session, confirm it remembers
streamlit run streamlit_app.py   # chat UI version of the same agent
```

### Streamlit UI

`streamlit run streamlit_app.py` opens a browser chat interface around
the same `Agent` class -- no changes to `agent.py` needed. Sidebar
shows everything currently stored in memory (with its source tag,
which becomes useful once Gate v1 exists) and lets you start a fresh
UI session (new chat history, same persistent memory) to test recall
the same way `agent.py`'s two-session demo does. A "wipe all memory"
button is there for resetting between manual tests.

The `agent.py` demo simulates two sessions: it tells the agent a fact
in "Session 1", then creates a brand-new `Agent` object for "Session
2" (same on-disk memory store) and confirms the fact carries over —
this is the exact mechanism the memory-injection attack in Phase 1
will exploit, just with a poisoned document instead of a genuine fact.

## Note on message format

Groq's API is OpenAI-compatible: messages use `{"role": "user"/"assistant", "content": "..."}`,
and the system prompt is passed as a `{"role": "system", ...}` message
at the start of the list rather than a separate parameter. This is
already handled inside `agent.py`'s `chat()` method.

## Note on the vector store's embedding model

`memory_store.py` uses Chroma's default embedding function, which
downloads a small ONNX model (~90MB) the first time you run it. This
requires normal internet access and works out of the box on your own
machine — no API key needed for this part, it's a local model.

## What's next (not built yet)

- `attacks/direct_injection.py`, `attacks/indirect_injection.py` —
  Phase 1's attack scripts, using `agent.process_document()`
- `gate.py` — Gate v1 (provenance tag + risk classification + routing),
  inserted at the `# TODO(Phase 1)` marker in `agent.py`'s `_write_path`
- `run_trials.py` — runs N trials of an attack, logs pass/fail, computes ASR