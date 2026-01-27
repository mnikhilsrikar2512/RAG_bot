# RAG_bot
# RAG Bot – Local Document Q&A

A fast, fully local Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from your documents.

Built with:
- LangChain  
- FAISS (vector store)  
- HuggingFace embeddings  
- Ollama (Mistral)  
- Gradio UI  

## Features
- Answers only from uploaded documents  
- Chat-friendly follow-ups (“elaborate”, “tell me more”, etc.)  
- Runs fully offline  
- Sub-second responses  

## Setup

```bash
git clone https://github.com/mnikhilsrikar2512/RAG_bot.git
cd RAG_bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
