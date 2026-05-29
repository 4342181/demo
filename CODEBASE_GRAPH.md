# Codebase Knowledge Graph
> Read this file once at session start. Navigate nodes by ID instead of re-reading source files.

---

## PROJECT OVERVIEW

**App Name:** Cognito AI  
**Purpose:** RAG (Retrieval-Augmented Generation) chatbot for RFP/RFQ document analysis, built for a sales team. Users upload documents (PDF, DOCX, images, etc.), ask questions, and get AI-powered answers drawn from a knowledge base.

**Stack:**
- Backend: Python · FastAPI · Azure OpenAI · Azure Document Intelligence · Azure Table Storage · Azure Blob Storage
- Frontend: React (CRA) · React Router · Axios · ReactMarkdown

---

## FILE MAP

| ID | Path | Role |
|----|------|------|
| F1 | `app.py` | FastAPI server — HTTP API, auth, RAG endpoint |
| F2 | `pipeline.py` | RAG engine — chunking, embedding, retrieval, LLM call |
| F3 | `main.py` | One-shot CLI script — builds knowledge base from Azure Blob, then runs interactive chat |
| F4 | `knowledge_base.pkl` | Persisted vector store (pickle) — list of `{text, embedding, source}` dicts |
| F5 | `Frontend/src/App.js` | Entire React app — Login, ChatAppContent, App (router), AppWithRouter |
| F6 | `Frontend/src/index.js` | React entry point — renders `<AppWithRouter />` |
| F7 | `Frontend/package.json` | Frontend deps: react, axios, react-router-dom, react-markdown, remark-gfm |
| F8 | `requirements.txt` | Backend deps |

---

## COMPONENT / FUNCTION GRAPH

### Backend (Python)
