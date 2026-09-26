"""
streamlit_app.py

A chat UI around agent.py -- type a message, see the reply, watch
memory persist across turns, all in a browser instead of the terminal.

Includes a Gate toggle (Off / Gate v1) so you can demo the SAME
poisoned document with the gate off vs. on, side by side, without
restarting the app. Gate decisions are shown live, color-coded:
green = auto_write, amber = quarantine, red = block.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from agent import Agent

st.set_page_config(page_title="Memory Agent Demo", page_icon="🧠", layout="centered")

st.title("🧠 Memory Agent Demo")
st.caption(
    "Chat with the agent below. It writes memory-worthy facts to a persistent "
    "vector store and retrieves them on later turns -- try telling it something, "
    "then asking about it in a later message."
)

GATE_LABELS = {"Off (ungated baseline)": None, "Gate v1": "v1", "Gate v2 (origin binding)": "v2"}

# --- Session state setup -------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = Agent(memory_path="./chroma_data_streamlit", session_id="ui-session-1", gate_version=None)
    st.session_state.messages = []
    st.session_state.session_count = 1
    st.session_state.gate_choice = "Off (ungated baseline)"

with st.sidebar:
    st.subheader("Gate")
    gate_choice = st.radio(
        "Write-path defense",
        options=list(GATE_LABELS.keys()),
        index=list(GATE_LABELS.keys()).index(st.session_state.gate_choice),
        help="Off reproduces the original vulnerable baseline. Gate v1 adds "
             "provenance + risk-classification checks before every write.",
    )
    st.session_state.gate_choice = gate_choice
    # Cheap to set every rerun -- gate_version is just an attribute read
    # inside _write_path, so flipping it doesn't require a new Agent
    # object or losing memory/session state.
    st.session_state.agent.gate_version = GATE_LABELS[gate_choice]

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
        st.session_state.agent = Agent(
            memory_path="./chroma_data_streamlit",
            session_id=f"ui-session-{st.session_state.session_count}",
            gate_version=GATE_LABELS[st.session_state.gate_choice],
        )
        st.session_state.messages = []
        st.rerun()

    if st.button("Wipe all memory"):
        # reset_all_memory() clears long-term store, quarantine store,
        # AND short-term history + gate log in one call.
        st.session_state.agent.reset_all_memory()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("What's currently stored")
    stored = st.session_state.agent.memory.all_memories()
    if stored:
        for m in stored:
            source = m["metadata"].get("source", "unknown")
            origin = m["metadata"].get("origin")
            origin_tag = f" · origin: {origin}" if origin else ""
            st.markdown(f"- *{m['text']}*  \n  `source: {source}{origin_tag}`")
    else:
        st.caption("Nothing stored yet.")

    quarantined = st.session_state.agent.quarantine.all_memories()
    if quarantined:
        st.subheader("Quarantined (held, not retrievable)")
        for m in quarantined:
            source = m["metadata"].get("source", "unknown")
            st.warning(f"{m['text']}\n\nsource: {source}", icon="🟡")

    if st.session_state.agent.gate_log:
        st.divider()
        st.subheader("Gate decisions this session")
        # Most recent first, color-coded by action.
        for entry in reversed(st.session_state.agent.gate_log[-8:]):
            label = f"**{entry['action']}** — {entry['text']}\n\nsource: {entry['source']} · risk: {entry['risk']}"
            if entry["action"] == "auto_write":
                st.success(label, icon="🟢")
            elif entry["action"] == "quarantine":
                st.warning(label, icon="🟡")
            elif entry["action"] == "block":
                st.error(label, icon="🔴")

# --- Chat history display -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

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
                    reply = st.session_state.agent.process_document(
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
                reply = st.session_state.agent.chat(user_text)
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()  # refresh sidebar's "what's stored" list immediately