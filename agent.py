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
import json
import time
from dotenv import load_dotenv
from groq import Groq, APIStatusError

from memory_store import MemoryStore
from extraction import extract_memory, MODEL as NOTE_MODEL
import gate as gate_v1
import provenance
import crypto_layer
import act_gate
from act_gate import UserAuthorizer  # act_gate is shadowed by a parameter name in __init__

load_dotenv()  # reads .env in the current folder and loads it into os.environ

# Groq deprecated llama-3.3-70b-versatile in June 2026 -- openai/gpt-oss-120b
# is their official recommended replacement for higher-quality chat replies.
MODEL = "openai/gpt-oss-120b"

# The agent's own note-taking (save_session_note) uses the cheaper
# extraction model, as memory frameworks like Mem0 do for
# consolidation -- and so it doesn't spend the chat model's quota.
NOTE_PROMPT = (
    "You are the note-taking module of an assistant with long-term memory. From the conversation "
    "above, write ONE short note, in your own words, of anything about the user worth remembering "
    "for future sessions (contacts, preferences, standing arrangements). Reply with only the note, "
    "or NONE if there is nothing worth keeping."
)


class Agent:
    def __init__(self, memory_path: str = "./chroma_data", session_id: str = "default", gate_version: str | None = None,
                 tools: bool = False, act_gate: bool | None = None,
                 paste_split: bool = True, k: int = 2, elevation: bool = True, propagation: bool = True):
        """
        gate_version: None -> ungated baseline (Phase 1's vulnerable
        agent, unchanged). "v1" -> Gate v1 active (provenance + risk
        classification + routing). "v2" -> origin binding + propagation
        (provenance.py), then Gate v1's risk routing on the bound origin.

        tools: offer the model the (simulated) send_email tool. Off by
        default so the original text-only scenarios run exactly as
        before -- offering a tool changes how the model answers.

        act_gate: run act_gate.py's Algorithm 1 on every tool call.
        Defaults to on for v2 and off otherwise; pass it explicitly
        for ablations (e.g. v2 write gate without the act gate).

        Ablation switches (defaults = the full design):
          paste_split -- v2 splits pasted material out of user_direct
                         messages (off: whole message keeps the channel label)
          k           -- independent trusted domains needed to elevate
                         an untrusted value (k=1 is the insecure single-source point)
          propagation -- M2: derived items (agent notes, trusted-tool
                         output) inherit the lowest origin of their inputs
                         (off: they keep the label of whoever wrote them)
          elevation   -- consult the trusted registries at all (off:
                         every untrusted value needs user confirmation)
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
        self.tools = tools
        self.act_gate_enabled = (gate_version == "v2") if act_gate is None else act_gate
        self.paste_split = paste_split
        self.k = k
        self.elevation = elevation
        self.propagation = propagation
        # v2: every origin that has fed this session's conversation so
        # far. The agent's own notes are derived from the whole
        # conversation, so they inherit the lowest of these (M2).
        self.session_origins: list[str] = []
        # Every tool call the model proposed, with the act gate's verdict
        # and whether it executed. outbox = calls that actually executed.
        self.action_log: list[dict] = []
        self.outbox: list[dict] = []
        # Calls the act gate HELD for a one-time user confirmation.
        self.pending_actions: list[dict] = []
        self.authorizer = UserAuthorizer()
        # v2 only (step 4): memory provenance is HMAC-signed, and every
        # write / verdict / confirmation goes into a hash-chained audit
        # log next to the store. Baselines stay untouched.
        self._signing_key = crypto_layer.load_key(memory_path) if gate_version == "v2" else None
        self.audit = (crypto_layer.AuditLog(os.path.join(memory_path, "audit_log.jsonl"), self._signing_key)
                      if gate_version == "v2" else None)

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
            return self._commit_v1(extracted, source_label)

        raise ValueError(f"Unknown gate_version: {self.gate_version!r}")

    def _commit_v1(self, extracted: str, source_label: str) -> str:
        decision = gate_v1.evaluate(extracted, source_label)
        self.gate_log.append({"text": extracted, "source": source_label, **decision.to_dict()})

        if decision.action == "auto_write":
            self.memory.write(extracted, metadata={"source": source_label, "gate_action": decision.action})
        elif decision.action == "quarantine":
            self.quarantine.write(extracted, metadata={"source": source_label, "gate_action": decision.action, "risk": decision.risk})
        elif decision.action == "block":
            pass  # not written anywhere; only logged in self.gate_log
        return extracted

    def _audit(self, event: str, data: dict):
        if self.audit is not None:
            self.audit.append(event, data)

    def _memory_origin(self, m: dict) -> str:
        # Memories written before v2 have no origin field -- derive it
        # from their channel label (fail-safe: unknown -> untrusted).
        text, meta = m.get("text", ""), m.get("metadata") or {}
        if "origin" not in meta:
            return provenance.channel_origin(meta.get("source", ""))
        # Under v2 an origin label only counts if its signature checks
        # out; a store edited to relabel a memory "user" is untrusted.
        if self._signing_key is not None and not crypto_layer.verify_memory(self._signing_key, text, meta):
            self._audit("signature_invalid", {"text": text[:200], "claimed_origin": meta.get("origin")})
            return "untrusted_external"
        return meta["origin"]

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
        if source_label == "user_direct" and self.paste_split:
            segments = provenance.split_message(source_text)
        else:
            origin = provenance.channel_origin(source_label)
            segments = [provenance.Segment(source_text, origin, f"channel '{source_label}'")]

        self.turn_origins = [s.origin for s in segments] + [self._memory_origin(m) for m in self._last_retrieved]
        self.session_origins.extend(self.turn_origins)

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
        return self._write_derived(text, "agent_note", self.session_origins, own_origin="agent")

    def save_session_note(self) -> str | None:
        """
        The agent summarizes this session into its own note and saves
        it (the self-summarization laundering channel, L-a). The note
        is written in the agent's words, so a poisoned instruction can
        come out looking like a harmless fact.
        """
        messages = self.conversation_history + [{"role": "user", "content": NOTE_PROMPT}]
        last_error = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(model=NOTE_MODEL, messages=messages, temperature=0)
                break
            except APIStatusError as e:
                if e.status_code == 429 or 500 <= e.status_code < 600:
                    last_error = e
                    time.sleep(2 ** attempt)
                else:
                    raise
        else:
            raise last_error
        note = (response.choices[0].message.content or "").strip()
        if not note or note.upper().rstrip(".") == "NONE":
            return None
        self.write_agent_note(note)
        return note

    def ingest_via_trusted_tool(self, tool_name: str, tool_fn, input_text: str, input_origin: str) -> list[str]:
        """
        Run a TRUSTED tool (e.g. a CRM importer) on some input and save
        what it returns. The output arrives on the trusted_tool
        channel -- but if the input was untrusted, the output is just
        the attacker's content echoed back (L-b). Under v2 the output
        inherits the input's untrust at the call boundary (M2).

        Records are written as-is, without the LLM extraction step: an
        import writes the records it produced, it doesn't ask a model
        whether they're worth remembering. (Routing them through
        extraction dropped 29/30 records in the first gate_v1 run, so
        the attack never reached any gate.)
        """
        records = tool_fn(input_text)
        self._audit("trusted_tool_call", {"tool": tool_name, "input_origin": input_origin, "records": records})
        for record in records:
            self._write_derived(record, "trusted_tool", [input_origin], own_origin="trusted_tool", extract=False)
        return records

    def _write_derived(self, text: str, source_label: str, input_origins: list[str], own_origin: str,
                       extract: bool = True):
        """
        Write path for content the agent or a tool DERIVED from other
        inputs. Baselines only see the channel label, so this is just
        the normal write path for them. Under v2 the origin is the
        lowest of (inputs + the writer's own label) when propagation
        is on (M2), or just the writer's label when it's off.

        extract=False writes `text` as the candidate memory directly
        (structured tool output) instead of running extraction on it.
        """
        if extract:
            if self.gate_version != "v2":
                return self._write_path(text, source_label)
            extracted = extract_memory(text)
            if not extracted:
                return None
        else:
            extracted = text
            if self.gate_version is None:
                self.memory.write(extracted, metadata={"source": source_label})
                return extracted
            if self.gate_version == "v1":
                return self._commit_v1(extracted, source_label)
        if self.propagation:
            origin = provenance.derive_origin(list(input_origins) + [own_origin])
            reason = f"{source_label} derived from {sorted(set(input_origins))}"
        else:
            origin, reason = own_origin, f"{source_label} (propagation off)"
        return self._commit_v2(extracted, origin, source_label, reason)

    def _commit_v2(self, text: str, origin: str, source_label: str, reason: str) -> str:
        act = provenance.act_class(origin)
        decision = gate_v1.evaluate(text, origin)
        meta = {"source": source_label, "origin": origin, "act_class": act, "gate_action": decision.action}
        if self._signing_key is not None:
            meta["sig"] = crypto_layer.sign_memory(self._signing_key, text, meta)
        self._audit("memory_write", {"text": text, "source": source_label, "origin": origin, "act_class": act,
                                     "action": decision.action, "risk": decision.risk, "provenance": reason})
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
        self.session_origins = []

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
        # The audit log is append-only: a reset is recorded, not erased.
        self._audit("memory_reset", {})
        self.gate_log = []
        self.action_log = []
        self.outbox = []
        self.pending_actions = []

    def _act_context(self, user_message: str, source_label: str) -> list:
        """
        What the act gate sees: this turn's message segments and the
        memories retrieved into this turn's prompt, each with its bound
        origin. Only v2 splits pasted material out of a user_direct
        message; without v2 the whole message carries its channel label.
        """
        if source_label == "user_direct" and self.gate_version == "v2" and self.paste_split:
            segments = provenance.split_message(user_message)
        else:
            segments = [provenance.Segment(user_message, provenance.channel_origin(source_label), "channel")]
        context = [
            act_gate.ContextItem(s.text, s.origin, "user_segment" if s.origin == "user" else "pasted_segment")
            for s in segments
        ]
        context += [act_gate.ContextItem(m["text"], self._memory_origin(m), "memory") for m in self._last_retrieved]
        return context

    def _dispatch_tool_call(self, name: str, args: dict, user_message: str, source_label: str) -> str:
        """
        Act gate -> (simulated) execution. Nothing is really sent; an
        executed call is appended to self.outbox, which is what the
        attack harness checks. Returns a short line for the reply.
        """
        verdict = None
        context = self._act_context(user_message, source_label)
        if self.act_gate_enabled:
            verdict = act_gate.authorize(name, args, context, authorizer=self.authorizer, **self._act_gate_kwargs())
        known = name in act_gate.SECURITY_FIELDS
        executed = known and (verdict is None or verdict.allowed)

        entry = {"tool": name, "args": args, "executed": executed, "verdict": verdict.to_dict() if verdict else None}
        self.action_log.append(entry)
        self._audit("tool_call", {"tool": name, "args": args, "executed": executed,
                                  "verdict": entry["verdict"]})
        recipients = f"to={args.get('to')} cc={args.get('cc') or []}"

        if executed:
            self.outbox.append({"tool": name, "args": args})
            return f"[{name} sent: {recipients}]"
        if known and verdict is not None and verdict.status == "needs_confirmation":
            pending_id = len(self.pending_actions)
            self.pending_actions.append({"id": pending_id, "tool": name, "args": args, "context": context,
                                         "verdict": verdict, "state": "pending"})
            return (f"[{name} HELD for your confirmation (#{pending_id}): {recipients} -- "
                    f"{verdict.unlicensed_values} came from an untrusted source and is not independently verified]")
        return f"[{name}: unknown tool, not executed]"

    def _act_gate_kwargs(self) -> dict:
        kwargs = {"k": self.k}
        if not self.elevation:
            kwargs["registries"] = []  # no trusted corroborators at all
        return kwargs

    def confirm_pending(self, pending_id: int) -> str:
        """
        The USER approves one held action (UI Confirm button, or the
        harness's simulated user). This is the only place a user
        authorization token is issued: bound to exactly this call,
        single-use, short-lived, then immediately presented to the
        act gate. The model has no tool that reaches this method.
        """
        p = self.pending_actions[pending_id]
        if p["state"] != "pending":
            return f"[#{pending_id} already {p['state']}]"
        token = self.authorizer.issue(p["tool"], p["args"])
        verdict = act_gate.authorize(p["tool"], p["args"], p["context"], authorizer=self.authorizer, token=token,
                                     **self._act_gate_kwargs())
        executed = verdict.allowed
        p["state"] = "confirmed" if executed else "failed"
        self._audit("user_confirmation", {"pending_id": pending_id, "tool": p["tool"], "args": p["args"],
                                          "executed": executed, "reason": verdict.reason})
        self.action_log.append({"tool": p["tool"], "args": p["args"], "executed": executed,
                                "verdict": verdict.to_dict(), "pending_id": pending_id})
        if executed:
            self.outbox.append({"tool": p["tool"], "args": p["args"], "confirmed_by_user": True})
            return f"[{p['tool']} sent after your confirmation: to={p['args'].get('to')} cc={p['args'].get('cc') or []}]"
        return f"[{p['tool']} still not authorized: {verdict.reason}]"

    def reject_pending(self, pending_id: int) -> str:
        p = self.pending_actions[pending_id]
        if p["state"] == "pending":
            p["state"] = "rejected"
            self._audit("user_rejection", {"pending_id": pending_id, "tool": p["tool"], "args": p["args"]})
        return f"[#{pending_id} {p['state']}]"

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
                    **({"tools": act_gate.TOOLS, "tool_choice": "auto"} if self.tools else {}),
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

        message = response.choices[0].message
        reply = message.content or ""
        # Tool calls go through the act gate. To save quota the tool
        # result is not sent back for a second model call; the outcome
        # is appended to the reply text instead.
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            outcome = self._dispatch_tool_call(call.function.name, args, user_message, source_label)
            reply = f"{reply}\n{outcome}".strip()
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