"""
streamlit_app.py

A minimal chat UI around agent.py -- lets you type a message, see the
reply, ask a follow-up, and watch memory persist across turns, all in
a browser instead of the terminal.

This does NOT change agent.py at all. It just wraps the existing
Agent class. Once Gate v1/v2 exist, this same UI can be extended to
show the "ungated vs. gated" side-by-side view mentioned in your
implementation plan -- for now it's just a clean way to interact with
what you already built.

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

# --- Session state setup -------------------------------------------------
# `st.session_state` persists across reruns WITHIN one browser tab/session,
# but a real Python process restart (or clicking "New session" below)
# creates a brand new Agent object -- same idea as agent1 / agent2 in
# agent.py's __main__ demo, just triggered from the UI instead of code.

if "agent" not in st.session_state:
    st.session_state.agent = Agent(memory_path="./chroma_data_streamlit", session_id="ui-session-1")
    st.session_state.messages = []  # what's shown on screen this session
    st.session_state.session_count = 1

with st.sidebar:
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
        )
        st.session_state.messages = []
        st.rerun()

    if st.button("Wipe all memory"):
        st.session_state.agent.memory.reset()       # long-term (vector store)
        st.session_state.agent.clear_short_term()    # short-term (chat history)
        st.session_state.messages = []               # what's displayed on screen
        st.rerun()

    st.divider()
    st.subheader("What's currently stored")
    stored = st.session_state.agent.memory.all_memories()
    if stored:
        for m in stored:
            source = m["metadata"].get("source", "unknown")
            st.markdown(f"- *{m['text']}*  \n  `source: {source}`")
    else:
        st.caption("Nothing stored yet.")

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