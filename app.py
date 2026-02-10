import gradio as gr
import time
from rag_core import hybrid_search, generate_answer

def gradio_rag(query):
    start = time.time()

    chunks = hybrid_search(query, top_k=15, rerank_k=20)
    answer, _ = generate_answer(query, top_k=3)

    end = time.time()
    response_time = end - start

    out = f"## Answer\n{answer}\n\n"
    out += f"**Response Time:** {response_time:.2f} seconds\n\n"
    out += "---\n"
    out += "## Retrieved Chunks\n"

    for c in chunks:
            out += f"**URL:** {c['url']}\n"
            out += f"- Dense score: {c.get('dense_score', 'N/A')}\n"
            out += f"- BM25 score: {c.get('bm25_score', 'N/A')}\n"
            out += f"- RRF score: {c.get('rrf_score', 'N/A')}\n"
            out += f"- Rerank score: {c.get('rerank_score', 'N/A')}\n\n"
            out += f"{c['text'][:600]}...\n\n"
            out += "---\n"

    return out

css = """
.gradio-container {max-width: 850px !important; margin: auto !important;}
"""
app = gr.Interface(
    fn=gradio_rag,
    inputs=gr.Textbox(
        lines=1,
        placeholder="Ask a question...",
        label="Query"
    ),
    outputs=gr.Markdown(label="Results"),
    title="Hybrid RAG System – Group 61",
    theme="glass",
    description="A minimal hybrid RAG demo using Dense + BM25 + RRF + Reranking.",
    flagging_mode="never",
    css=css
)

app.launch(debug=True)