"""
streamlit_app.py

A chat UI around agent.py -- type a message, see the reply, watch
memory persist across turns, all in a browser instead of the terminal.

Includes a Gate toggle (Off / Gate v1 / Gate v2) so you can demo the
SAME poisoned document with the gate off vs. on, side by side, without
restarting the app. Gate decisions are shown live, color-coded:
green = auto_write, amber = quarantine, red = block.

With the send_email tool on, the agent can (simulated) send email.
Under Gate v2 a call whose recipients came from an untrusted source
and aren't independently verified is HELD: it appears above the chat
box with where the value came from, and runs only if you press
Confirm (a single-use token bound to exactly that call). Nothing is
really sent -- executed calls are listed in the sidebar's outbox.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from agent import Agent

MEMORY_PATH = "./chroma_data_streamlit"

st.set_page_config(page_title="Memory Agent Demo", page_icon="🧠", layout="centered")

st.title("🧠 Memory Agent Demo")
st.caption(
    "Chat with the agent below. It writes memory-worthy facts to a persistent "
    "vector store and retrieves them on later turns -- try telling it something, "
    "then asking about it in a later message."
)

GATE_LABELS = {"Off (ungated baseline)": None, "Gate v1": "v1", "Gate v2 (origin binding)": "v2"}


def make_agent(gate_version, tools: bool, session_count: int, history: list | None = None) -> Agent:
    agent = Agent(memory_path=MEMORY_PATH, session_id=f"ui-session-{session_count}",
                  gate_version=gate_version, tools=tools)
    if history:
        agent.conversation_history = list(history)
    return agent


# --- Session state setup -------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.session_count = 1
    st.session_state.gate_choice = "Off (ungated baseline)"
    st.session_state.tools_on = True
    st.session_state.agent = make_agent(None, True, 1)
    st.session_state.agent_config = (None, True)
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Gate")
    gate_choice = st.radio(
        "Write-path defense",
        options=list(GATE_LABELS.keys()),
        index=list(GATE_LABELS.keys()).index(st.session_state.gate_choice),
        help="Off reproduces the original vulnerable baseline. Gate v1 adds "
             "provenance + risk-classification checks before every write. Gate v2 "
             "binds origin at write time and checks every action (act gate).",
    )
    tools_on = st.toggle(
        "Give the agent the send_email tool",
        value=st.session_state.tools_on,
        help="Simulated -- nothing is really sent. Needed for the act gate to have "
             "anything to check.",
    )
    st.session_state.gate_choice = gate_choice
    st.session_state.tools_on = tools_on

    # Gate v2's act gate, audit log and signing key are set up when the
    # Agent is created, so a changed setting needs a new Agent (same
    # on-disk memory; the chat so far is carried over).
    config = (GATE_LABELS[gate_choice], tools_on)
    if config != st.session_state.agent_config:
        st.session_state.agent = make_agent(*config, st.session_state.session_count,
                                            history=st.session_state.agent.conversation_history)
        st.session_state.agent_config = config

    agent = st.session_state.agent

    st.divider()
    st.subheader("Session controls")
    st.write(f"Current session: **ui-session-{st.session_state.session_count}**")
    st.caption(
        "Long-term memory persists across sessions (same on-disk store). "
        "Short-term chat history does not -- starting a new session simulates "
        "opening a fresh conversation with the same agent."
    )

    if st.button("Start new session (keep memory)"):
        st.session_state.session_count += 1
        st.session_state.agent = make_agent(*st.session_state.agent_config, st.session_state.session_count)
        st.session_state.messages = []
        st.rerun()

    if st.button("Wipe all memory"):
        # reset_all_memory() clears long-term store, quarantine store,
        # short-term history, gate log, and held / sent actions.
        agent.reset_all_memory()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("What's currently stored")
    stored = agent.memory.all_memories()
    if stored:
        for m in stored:
            source = m["metadata"].get("source", "unknown")
            origin = m["metadata"].get("origin")
            origin_tag = f" · origin: {origin}" if origin else ""
            st.markdown(f"- *{m['text']}*  \n  `source: {source}{origin_tag}`")
    else:
        st.caption("Nothing stored yet.")

    quarantined = agent.quarantine.all_memories()
    if quarantined:
        st.subheader("Quarantined (held, not retrievable)")
        for m in quarantined:
            source = m["metadata"].get("source", "unknown")
            st.warning(f"{m['text']}\n\nsource: {source}", icon="🟡")

    if agent.gate_log:
        st.divider()
        st.subheader("Gate decisions this session")
        # Most recent first, color-coded by action.
        for entry in reversed(agent.gate_log[-8:]):
            origin = f" · origin: {entry['origin']}" if entry.get("origin") else ""
            label = f"**{entry['action']}** — {entry['text']}\n\nsource: {entry['source']}{origin} · risk: {entry['risk']}"
            if entry["action"] == "auto_write":
                st.success(label, icon="🟢")
            elif entry["action"] == "quarantine":
                st.warning(label, icon="🟡")
            elif entry["action"] == "block":
                st.error(label, icon="🔴")

    if agent.outbox:
        st.divider()
        st.subheader("Outbox (simulated, this session)")
        for sent in reversed(agent.outbox):
            args = sent["args"]
            how = " · confirmed by you" if sent.get("confirmed_by_user") else ""
            st.markdown(f"- **{args.get('subject', '(no subject)')}**  \n"
                        f"  to: `{', '.join(args.get('to') or [])}`  \n"
                        f"  cc: `{', '.join(args.get('cc') or []) or '—'}`{how}")

# --- Chat history display -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Actions held for confirmation ----------------------------------------
# Only this UI calls confirm_pending / reject_pending: the model has no
# tool that reaches them, so it can't approve its own held actions.
pending = [p for p in agent.pending_actions if p["state"] == "pending"]
if pending:
    st.subheader("⏸️ Waiting for your confirmation")
    for p in pending:
        args, verdict = p["args"], p["verdict"]
        with st.container(border=True):
            st.markdown(f"**{p['tool']}** #{p['id']} — *{args.get('subject', '(no subject)')}*")
            st.markdown(f"to: `{', '.join(args.get('to') or [])}`  \n"
                        f"cc: `{', '.join(args.get('cc') or []) or '—'}`")
            st.warning(
                f"Not independently verified: **{', '.join(verdict.unlicensed_values)}**. "
                "It came from:\n\n" + "\n".join(f"- {src}" for src in verdict.tainted_by),
                icon="⚠️",
            )
            if args.get("body"):
                with st.expander("Email body"):
                    st.write(args["body"])
            confirm_col, reject_col = st.columns(2)
            if confirm_col.button("✅ Confirm and send", key=f"confirm_{p['id']}", use_container_width=True):
                result = agent.confirm_pending(p["id"])
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Confirmed #{p['id']}. {result}"})
                st.rerun()
            if reject_col.button("🚫 Reject", key=f"reject_{p['id']}", use_container_width=True):
                agent.reject_pending(p["id"])
                st.session_state.messages.append({"role": "assistant", "content": f"🚫 Rejected #{p['id']} -- not sent."})
                st.rerun()

# --- Chat input (text + optional file attachment, same box) ---------------
# accept_file=True adds a paperclip attach button right inside the chat
# box -- attaching a file simulates the agent processing a document/email
# as part of a task (the same entry point attacks/indirect_injection.py
# uses), while typing alone behaves exactly like before.

submission = st.chat_input(
    "Type a message, or attach a document to have the agent process it...",
    accept_file=True,
    file_type=["txt", "md"],
)

if submission:
    user_text = submission.text
    attached_files = submission.files  # list, empty if no file attached

    if attached_files:
        # One or more files attached -- process each via the write path
        # agent.process_document() uses (source_label="processed_document").
        # If the user also typed text, that's used as the task instruction;
        # otherwise fall back to a generic one.
        task_instruction = user_text.strip() or "Please summarize this document for me."

        for f in attached_files:
            document_text = f.read().decode("utf-8", errors="replace")

            display_text = f"📄 **{f.name}** — task: \"{task_instruction}\""
            st.session_state.messages.append({"role": "user", "content": display_text})
            with st.chat_message("user"):
                st.write(display_text)

            with st.chat_message("assistant"):
                with st.spinner(f"Processing {f.name}..."):
                    reply = agent.process_document(
                        document_text=document_text,
                        task_instruction=task_instruction,
                        source_label="processed_document",
                    )
                st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    elif user_text.strip():
        # Plain text message -- identical to the original chat behavior.
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = agent.chat(user_text)
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()  # refresh sidebar + pending actions immediately
