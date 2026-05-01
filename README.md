<img width="1439" height="813" alt="Снимок экрана 2026-05-01 в 06 03 45" src="https://github.com/user-attachments/assets/30d07ec6-6276-4f7b-8f44-f802824fde90" />
<img width="1440" height="822" alt="Снимок экрана 2026-05-01 в 06 03 09" src="https://github.com/user-attachments/assets/4966c2e0-d9c7-4285-a82e-12bed1f95e1b" />









# 🌌 Quantum RAG Chatbot (Multi-University AI)

An elegant and incredibly powerful AI assistant built on a **RAG (Retrieval-Augmented Generation)** architecture. This project is designed for intelligent search and answer generation based on Turkish university databases. 

It combines a blazing-fast Python backend with a premium Spatial UI, written entirely without frontend frameworks.

## ✨ Key Features

* **Quantum Spatial UI:** A unique framework-less interface. Features multi-layered Glassmorphism, magnetic buttons with real physics, kinetic cursor tracking, and fluid animations.
* **Smart RAG Engine:** Automatic text chunking, vectorization via OpenAI's `text-embedding-3-small`, and high-quality response generation using `gpt-4o-mini`.
* **In-Memory Vector DB:** Utilizes **FAISS** for instant semantic search across thousands of JSON documents.
* **AI Thought Simulation:** Includes a visualizer for neural network processing phases (Chain of Thought) and smooth character-by-character response typing (Typewriter Effect).
* **Rich Markdown Support:** Beautiful rendering of lists, bold text, quotes, and Mac-style code blocks directly in the chat.
* **Multilingual & Themes:** Native support for Turkish and English (TR/EN), alongside flawless Dark and Light modes.

## 🛠 Tech Stack

* **Backend:** Python 3.11+, FastAPI, Uvicorn
* **AI & Vectors:** OpenAI API, FAISS, NumPy
* **Frontend:** Pure HTML5, CSS3, Vanilla JavaScript (Zero Dependencies)

## 🚀 Installation & Setup

To run this project locally, you will need Python 3.9 or higher.

**1. Clone the repository**
```bash
git clone [https://github.com/AbdurahmanKa2003/quantum-rag-chatbot.git](https://github.com/AbdurahmanKa2003/quantum-rag-chatbot.git)
cd quantum-rag-chatbot

python -m venv venv
source venv/bin/activate  # For macOS and Linux
# venv\Scripts\activate   # For Windows

pip install fastapi uvicorn openai faiss-cpu numpy pydantic python-dotenv

OPENAI_API_KEY=sk-your-secret-api-key-here

python chatbot_api.py
