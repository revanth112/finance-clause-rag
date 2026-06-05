import streamlit as st
import requests

st.set_page_config(
    page_title="Finance Clause RAG",
    page_icon="📄",
    layout="wide"
)

st.title("Finance Clause Intelligence Assistant")
st.caption("Powered by BGE-M3 + FAISS + Cross-Encoder Reranker + Azure GPT-4o-mini")

st.sidebar.header("Metadata Filters")
st.sidebar.markdown("Use filters to narrow document search before retrieval.")

doc_type = st.sidebar.selectbox(
    "Document Type",
    ["", "contract", "policy", "invoice", "sop", "agreement"]
)
vendor_name = st.sidebar.text_input("Vendor Name")
clause_type = st.sidebar.selectbox(
    "Clause Type",
    ["", "termination", "payment", "liability", "confidentiality", "indemnity", "compliance"]
)
jurisdiction = st.sidebar.text_input("Jurisdiction")
risk_level = st.sidebar.selectbox(
    "Risk Level",
    ["", "low", "medium", "high"]
)

st.divider()

question = st.text_area(
    "Ask a question about your finance documents",
    placeholder="e.g. What is the termination notice period for Vendor A?",
    height=100
)

if st.button("Search & Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            payload = {
                "question": question,
                "filters": {
                    "document_type": doc_type or None,
                    "vendor_name": vendor_name or None,
                    "clause_type": clause_type or None,
                    "jurisdiction": jurisdiction or None,
                    "risk_level": risk_level or None
                },
                "top_k": 5
            }
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query/",
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Answer")
                    st.write(data.get("answer", "No answer returned."))
                    st.divider()
                    st.subheader("Retrieved Sources")
                    for i, source in enumerate(data.get("sources", [])):
                        with st.expander(f"Source {i+1}: {source.get('section_title', 'Unknown')} | {source.get('clause_type', '')} | Score: {source.get('score', 0):.3f}"):
                            st.write(source.get("text", ""))
                            st.caption(f"Document: {source.get('document_name', '')} | Page: {source.get('page_number', '')} | Vendor: {source.get('vendor_name', '')}")
                else:
                    st.error(f"API error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

st.divider()
st.caption("Part 1 - UI skeleton ready. Full RAG pipeline integrated in Parts 2-5.")
