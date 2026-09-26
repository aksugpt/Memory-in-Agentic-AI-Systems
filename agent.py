"""
agent.py

The agent's chat loop, using memory_store.py for long-term memory and
extraction.py to decide what's memory-worthy.

Write path: Input -> Extraction -> Gate (if enabled) -> Formatting ->
Persistence. gate_version=None reproduces Phase 1's original
vulnerable baseline unchanged (writes everything unconditionally --
this is what the no_gate ASR trials measure). gate_version="v1"
routes through gate.py's provenance + risk-classification checkpoint.
gate_version="v2" adds provenance.py's TMA-NM-style origin binding
(M1) and non-malleable propagation (M2), including splitting pasted
material out of user_direct chat messages.

Uses Groq (free tier, far more headroom than Gemini's) for both chat
and extraction/classification.
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq, APIStatusError

from memory_store import MemoryStore
from extraction import extract_memory
import gate as gate_v1
import provenance

load_dotenv()  # reads .env in the current folder and loads it into os.environ

# Groq deprecated llama-3.3-70b-versatile in June 2026 -- openai/gpt-oss-120b
# is their official recommended replacement for higher-quality chat replies.
MODEL = "openai/gpt-oss-120b"


class Agent:
    def __init__(self, memory_path: str = "./chroma_data", session_id: str = "default", gate_version: str | None = None):
        """
        gate_version: None -> ungated baseline (Phase 1's vulnerable
        agent, unchanged). "v1" -> Gate v1 active (provenance + risk
        classification + routing). "v2" -> origin binding + propagation
        (provenance.py), then Gate v1's risk routing on the bound origin.
        """
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.memory = MemoryStore(path=memory_path, collection_name="agent_memory")
        # Quarantined items live in a SEPARATE collection, deliberately
        # never queried by memory.retrieve() -- "held, not yet
        # retrievable" per the write-gate design, not silently dropped.
        self.quarantine = MemoryStore(path=memory_path, collection_name="agent_memory_quarantine")
        self.session_id = session_id
        self.gate_version = gate_version
        # Log of every gate decision made this session -- useful for
        # debugging and for the Streamlit sidebar to show later.
        self.gate_log: list[dict] = []
        # Short-term memory for THIS session only -- cleared when the
        # Agent object is thrown away, simulating a fresh session.
        # Groq/OpenAI-style messages: {"role": "user"/"assistant", "content": "..."}
        self.conversation_history: list[dict] = []
        # v2 only: origins of everything that fed the current turn
        # (message segments + retrieved memories). Anything the agent
        # itself writes this turn inherits the lowest of these (M2).
        self.turn_origins: list[str] = []
        self._last_retrieved: list[dict] = []

    def _build_system_prompt(self, user_message: str) -> str:
        """
        Read path: retrieve relevant long-term memories and inject
        them into context. This is the RAG-like half of the system.
        """
        memories = self.memory.retrieve(user_message, k=3)
        self._last_retrieved = memories
        if not memories:
            return "You are a helpful assistant."

        memory_lines = "\n".join(f"- {m['text']}" for m in memories)
        return (
            "You are a helpful assistant. Here is what you remember "
            f"about the user from past interactions:\n{memory_lines}\n\n"
            "Use this naturally when relevant to answer the user's question."
        )

    def _write_path(self, source_text: str, source_label: str = "user_direct"):
        """
        Input -> Extraction -> Gate (if enabled) -> Write.

        With gate_version=None: identical to the original vulnerable
        baseline -- extraction's output is written unconditionally.
        This is what your no_gate ASR trials measure.

        With gate_version="v1": Gate v1 evaluates the extracted
        candidate BEFORE it's committed, based on source_label's trust
        tier and a risk classification of the content itself.
        """
        if self.gate_version == "v2":
            return self._write_path_v2(source_text, source_label)

        extracted = extract_memory(source_text)
        if not extracted:
            return None

        if self.gate_version is None:
            # Original vulnerable baseline -- no gate, write unconditionally.
            self.memory.write(extracted, metadata={"source": source_label})
            return extracted

        if self.gate_version == "v1":
            decision = gate_v1.evaluate(extracted, source_label)
            self.gate_log.append({"text": extracted, "source": source_label, **decision.to_dict()})

            if decision.action == "auto_write":
                self.memory.write(extracted, metadata={"source": source_label, "gate_action": decision.action})
            elif decision.action == "quarantine":
                self.quarantine.write(extracted, metadata={"source": source_label, "gate_action": decision.action, "risk": decision.risk})
            elif decision.action == "block":
                pass  # not written anywhere; only logged in self.gate_log
            return extracted

        raise ValueError(f"Unknown gate_version: {self.gate_version!r}")

    @staticmethod
    def _memory_origin(m: dict) -> str:
        # Memories written before v2 have no origin field -- derive it
        # from their channel label (fail-safe: unknown -> untrusted).
        meta = m.get("metadata") or {}
        return meta.get("origin") or provenance.channel_origin(meta.get("source", ""))

    def _write_path_v2(self, source_text: str, source_label: str):
        """
        Gate v2 write path (TMA-NM M1 + M2, small scale):

          1. Bind origin from the channel. A user_direct message is
             further split so pasted material is labelled
             untrusted_external rather than inheriting the user's trust.
          2. Extract from EACH segment separately, so a candidate
             memory is derived from exactly one input and its origin is
             derive_origin([that input]) -- a pasted "note to self"
             cannot borrow the user's label by sharing a message.
          3. act_class = f(origin), stored with the memory.
          4. Gate v1's risk routing runs on the bound origin, not on
             the raw channel label.
        """
        if source_label == "user_direct":
            segments = provenance.split_message(source_text)
        else:
            origin = provenance.channel_origin(source_label)
            segments = [provenance.Segment(source_text, origin, f"channel '{source_label}'")]

        self.turn_origins = [s.origin for s in segments] + [self._memory_origin(m) for m in self._last_retrieved]

        written = []
        for seg in segments:
            extracted = extract_memory(seg.text)
            if not extracted:
                continue
            origin = provenance.derive_origin([seg.origin])
            written.append(self._commit_v2(extracted, origin, source_label, seg.reason))
        return written or None

    def write_agent_note(self, text: str):
        """
        The agent writing its OWN note (e.g. a summary of what it just
        read). Under v2 the note inherits the lowest origin of
        everything that fed this turn (M2), so summarizing a poisoned
        document cannot launder it into a trusted memory (L-a).
        """
        if self.gate_version != "v2":
            self.memory.write(text, metadata={"source": "agent_note"})
            return
        origin = provenance.derive_origin(self.turn_origins)
        self._commit_v2(text, origin, "agent_note", f"derived from {sorted(set(self.turn_origins))}")

    def _commit_v2(self, text: str, origin: str, source_label: str, reason: str) -> str:
        act = provenance.act_class(origin)
        decision = gate_v1.evaluate(text, origin)
        meta = {"source": source_label, "origin": origin, "act_class": act, "gate_action": decision.action}
        self.gate_log.append({"text": text, "source": source_label, "origin": origin, "act_class": act,
                              "provenance": reason, **decision.to_dict()})

        if decision.action == "auto_write":
            self.memory.write(text, metadata=meta)
        elif decision.action == "quarantine":
            self.quarantine.write(text, metadata={**meta, "risk": decision.risk})
        # "block": not written anywhere; only logged in self.gate_log
        return text

    def clear_short_term(self):
        """
        Clears only the in-session chat history (short-term/working
        memory) -- does NOT touch either long-term store. Useful for
        "start a fresh conversation, keep what's been remembered" flows.
        """
        self.conversation_history = []

    def reset_all_memory(self):
        """
        True full reset: long-term store, quarantine store, AND
        short-term chat history. memory.reset() alone only clears the
        main collection -- quarantine is a separate collection and
        needs its own reset, easy to forget since it's a newer addition.
        """
        self.memory.reset()
        self.quarantine.reset()
        self.clear_short_term()
        self.gate_log = []

    def chat(self, user_message: str, source_label: str = "user_direct") -> str:
        """
        One turn: retrieve -> respond -> extract -> write.

        source_label defaults to "user_direct" for normal chat, but
        process_document() overrides it to "processed_document" so the
        SAME write_path call is correctly tagged, instead of writing
        twice under two different labels (that was a real bug -- see
        the note in process_document below).
        """
        system_prompt = self._build_system_prompt(user_message)

        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history

        last_error = None
        response = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                break
            except APIStatusError as e:
                if e.status_code == 429:
                    last_error = e
                    wait = 2 ** attempt
                    print(f"  [agent] Groq rate-limited (429), retrying in {wait}s...")
                    time.sleep(wait)
                elif 500 <= e.status_code < 600:
                    last_error = e
                    wait = 2 ** attempt
                    print(f"  [agent] Groq server error ({e.status_code}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise last_error

        reply = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": reply})

        # Write path runs on the user's own message by default, tagged
        # with whatever source_label the caller specified.
        self._write_path(user_message, source_label=source_label)

        return reply

    def process_document(self, document_text: str, task_instruction: str, source_label: str = "processed_document") -> str:
        """
        Simulates the agent being handed a document/email to work on
        (e.g. "summarize this email"). This is the entry point Phase
        1's indirect-injection attack uses: the document's content
        goes through extraction just like a user message would, but
        is tagged with source_label ("processed_document" by default)
        instead of "user_direct" -- this is what Gate v1 will read to
        tell trusted input apart from untrusted input.

        BUGFIX: this used to call self.chat() (which writes under
        "user_direct") AND a second, separate _write_path() call
        tagged "processed_document" -- writing the same poisoned fact
        TWICE under two different trust labels. One of those copies
        was silently mislabeled as trusted, which would have made
        Gate v1 look artificially weaker than it really is (since the
        mislabeled copy would sail through regardless of gate logic).
        Now there's exactly one write, correctly tagged.
        """
        full_input = f"{task_instruction}\n\n---\n{document_text}\n---"
        reply = self.chat(full_input, source_label=source_label)
        return reply


if __name__ == "__main__":
    # Manual sanity check -- requires GROQ_API_KEY to be set.
    # Simulates: tell the agent a fact in "session 1", start a fresh
    # Agent object ("session 2"), confirm it remembers.
    print("=== Session 1 ===")
    agent1 = Agent(memory_path="./chroma_data_demo", session_id="s1")
    agent1.memory.reset()
    print("User: My favorite color is blue.")
    print("Agent:", agent1.chat("My favorite color is blue."))

    print("\n=== Session 2 (fresh Agent object, same memory store) ===")
    agent2 = Agent(memory_path="./chroma_data_demo", session_id="s2")
    print("User: What's my favorite color?")
    print("Agent:", agent2.chat("What's my favorite color?"))