import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_PATH = "vector_db"

# Fast local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Ultra-fast local LLM via Ollama
llm = ChatOllama(model="mistral", temperature=0)

SYSTEM_PROMPT = """
You are an educational assistant.
Answer strictly from the provided context.
If the answer is not in the context, say:
"I could not find this in the provided documents."

End with a clear conclusion.
"""

def chat(message, history):
    if history is None:
        history = []

    # Get last user question
    last_user_q = None
    for item in reversed(history):
        if item["role"] == "user":
            last_user_q = item["content"]
            break

    followups = {
        "elaborate", "elaborate it", "explain more",
        "tell me more", "expand", "continue"
    }

    if last_user_q and message.lower().strip() in followups:
        effective_query = f"Explain in more detail: {last_user_q}"
    else:
        effective_query = message

    docs = retriever.invoke(effective_query)

    # If retriever finds nothing meaningful, broaden the query
    if not docs or sum(len(d.page_content) for d in docs) < 300:
        docs = retriever.invoke(f"Overview of {effective_query}")

    context = "\n".join(d.page_content[:250] for d in docs)

    prompt = f"""{SYSTEM_PROMPT}
Use a clear, conversational tone.

Context:
{context}

Question:
{effective_query}

Answer:
"""

    response = llm.invoke(prompt).content.strip()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, ""



with gr.Blocks(title="Educational RAG Bot") as demo:
    gr.Markdown("# 📚 Educational RAG Chatbot")

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Ask from your documents...")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(share=True)
