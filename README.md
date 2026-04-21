# RAG Bot – AI Chatbot with Contextual Intelligence

An AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses by combining semantic search with language models.

## 🚀 Features

- Context-aware question answering  
- Semantic search using embeddings  
- Vector database integration  
- LLM-based response generation  
- FastAPI backend for API access  

## 🧠 How It Works

1. User sends a query  
2. System converts query into embeddings  
3. Relevant data is retrieved from vector storage  
4. LLM generates a response using retrieved context  
5. Final answer is returned to the user  

## 🛠 Tech Stack

- Python  
- FastAPI  
- Embeddings / Vector Search  
- LLM Integration  

## Setup

```bash
git clone https://github.com/mnikhilsrikar2512/RAG_bot.git
cd RAG_bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
