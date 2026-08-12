# RAG Retrieval-Augmented Generation Chatbot

## Description

This project builds a complete **Retrieval-Augmented Generation (RAG) Chatbot** using **LangChain**, **Google Gemini LLM (`gemini-3.5-flash`)**, **Google Gemini Embeddings (`models/gemini-embedding-001`)**, and **ChromaDB**. 

Instead of relying solely on the general knowledge of an LLM, this chatbot retrieves relevant text chunks from a local knowledge base (`data/sample.txt`), injects them as grounded context into a prompt template, and generates accurate, factually grounded answers.

---

## Architecture

```text
                    YOUR DOCUMENTS
                          │
                          ▼
                     CHUNKING (800 chars)
                          │
                          ▼
                     EMBEDDINGS (Gemini 3072-dim)
                          │
                          ▼
                       CHROMADB (Persisted)
                          │
                          │
                     USER QUERY
                          │
                          ▼
                      RETRIEVER
                          │
                          ▼
                    TOP-K CHUNKS (K=4)
                          │
                          ▼
                     CONTEXT INJECTION
                          │
                          ▼
                   PROMPT TEMPLATE
                          │
                          ▼
                    GEMINI LLM (gemini-3.5-flash)
                          │
                          ▼
                   GROUNDED ANSWER
```

---

## Two-Phase RAG Flow

### 1. Indexing Phase (Document Ingestion)

The indexing script runs only when documents need to be added or re-indexed:

```text
Document (data/sample.txt)
   ↓
RecursiveCharacterTextSplitter (chunk_size=800, overlap=50)
   ↓
Google Gemini Embeddings (models/gemini-embedding-001)
   ↓
Persistent ChromaDB Vector Database (./chroma_db)
```

### 2. Querying Phase (Chatbot Interaction)

The chatbot loads the existing ChromaDB database without regenerating embeddings on every query:

```text
User Question
   ↓
LangChain Retriever (similarity search)
   ↓
Top-K Relevant Chunks
   ↓
Context Injection ("\n\n".join(chunks))
   ↓
ChatPromptTemplate (strict grounding rules)
   ↓
Google Gemini Chat Model (temperature=0)
   ↓
Grounded AI Answer + Retrieved Context Display
```

---

## Key Features

- **Grounded AI Answers**: The chatbot answers using ONLY retrieved document context.
- **Hallucination Prevention**: If the context does not contain the answer, the LLM safely responds: `"I don't have enough information in the provided documents to answer that question."`
- **Context Injection**: Combines retrieved document chunks into a dynamic context block passed to the prompt template.
- **Top-K Comparison Mode**: Easily compare answer quality and retrieved context for $K=2$, $K=4$, and $K=6$.
- **Retrieved Context Display**: Shows the exact source, chunk ID, and text snippet used to generate the answer.
- **Modern LangChain APIs**: Built using `ChatPromptTemplate`, `ChatGoogleGenerativeAI`, `Chroma`, and `retriever.invoke()`.
- **Persistent Vector Store**: Avoids re-indexing documents on every run.

---

## Concepts Explained

### 1. What is RAG?
Retrieval-Augmented Generation combines **information retrieval** with **language generation**. The retriever searches external documents for relevant facts, and the language model uses those facts to write an answer.

### 2. Context Injection
Context injection is inserting retrieved text snippets into the LLM prompt before sending the request:

```text
User Question + Retrieved Context ──► Prompt Template ──► Gemini LLM
```

```python
context = "\n\n".join(doc.page_content for doc in documents)
prompt = prompt_template.format(context=context, question=question)
```

### 3. Grounding vs. Normal LLM

| Feature | Normal LLM | RAG Chatbot |
| :--- | :--- | :--- |
| **Data Source** | Static pre-trained weights | Your private / custom documents |
| **Accuracy** | May hallucinate outdated facts | Factually grounded in retrieved evidence |
| **Updates** | Requires expensive retraining | Update `data/sample.txt` & re-index |

### 4. Similarity Search vs. Retriever
- `vectorstore.similarity_search(query, k=4)`: Performs vector similarity comparison directly against ChromaDB.
- `vectorstore.as_retriever(search_kwargs={"k": 4})`: Creates a standard LangChain `Retriever` interface (`retriever.invoke(query)`), which easily integrates into LangChain chains and pipelines.

### 5. Top-K Trade-offs
- **Small $K$ ($K=2$)**: Highly focused, low token cost, but might miss broader context.
- **Medium $K$ ($K=4$)**: Default balance of context depth and precision.
- **Large $K$ ($K=6$)**: Broad context coverage, but higher token usage and potential noise.

### 6. Hallucination Prevention
By setting `temperature=0` and utilizing a strict system prompt (`Answer using ONLY the provided context`), the LLM refrains from fabricating unsupported facts.

---

## Installation & Setup

1. **Clone & Create Virtual Environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

---

## Running the Chatbot

Launch the interactive CLI application:

```powershell
python main.py
```

### Main Menu Options:

```text
==================================================
RAG CHATBOT
==================================================
1. Ask Question
2. Compare Top-K
3. Exit
```

- **Option 1 (Ask Question)**: Continuous interactive Q&A session. Type `exit` to return to the main menu.
- **Option 2 (Compare Top-K)**: Compare $K=2$, $K=4$, and $K=6$ context chunks and AI answers side-by-side.
- **Option 3 (Exit)**: Quit the program.

---

## Testing Guide

Try these questions in the chatbot to test performance:

### Basic Knowledge Questions:
1. `What is artificial intelligence?`
2. `What is machine learning?`
3. `What are embeddings?`
4. `What is a vector database?`
5. `What is ChromaDB?`

### Retrieval & RAG Questions:
6. `What is Retrieval-Augmented Generation?`
7. `What is similarity search?`
8. `What is Top-K retrieval?`
9. `What is the difference between RAG and fine-tuning?`
10. `What is prompt injection?`

### Out-of-Scope / Hallucination Test:
11. `What is the population of Mars?`
   - *Expected Response*: `"I don't have enough information in the provided documents to answer that question."`

---
