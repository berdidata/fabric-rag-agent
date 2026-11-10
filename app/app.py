import streamlit as st

from rag import ask

st.set_page_config(page_title="Docs Q&A", page_icon="📄")
st.title("Ask the internal docs")

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    st.chat_message(m["role"]).write(m["content"])

if q := st.chat_input("Ask a question"):
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, cites = ask(q, st.session_state.history[-6:])
        st.write(answer)
        if cites:
            with st.expander("Sources"):
                for i, c in enumerate(cites, 1):
                    st.markdown(f"**[{i}] {c['source']}, p.{c['page']}**")
                    st.caption(c["content"][:400] + "...")
    st.session_state.history += [{"role": "user", "content": q},
                                 {"role": "assistant", "content": answer}]
