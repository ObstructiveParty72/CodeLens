 HEAD
#  Codebase RAG — AI Software Engineer

An AI-powered **Codebase Question Answering System** that allows developers to ask natural-language questions about an entire software repository.

Instead of manually searching through hundreds of files, developers can ask questions such as:

> **"Where is authentication implemented?"**

> **"How does the rate limiter work?"**

> **"Which files handle database connections?"**

> **"Explain the request flow from the API endpoint to the database."**

The system retrieves the most relevant parts of the codebase and uses an LLM to generate a context-aware answer.

---

##  Why Codebase RAG?

Large software repositories are difficult to understand because important information is distributed across:

* Multiple files
* Classes and functions
* Configuration files
* Dependencies
* Documentation
* Different modules

Traditional keyword search is often insufficient because a developer may not know the exact function or file name they are looking for.

**Codebase RAG combines code ingestion, embeddings, vector search, and LLM-based generation to create an AI assistant that understands a repository.**

---

##  Features

###  Repository Ingestion

Load and analyze a complete local or GitHub repository.

The ingestion pipeline:

```text
Repository
    ↓
File Discovery
    ↓
File Filtering
    ↓
Code Parsing
    ↓
Code Chunking
    ↓
Embedding Generation
    ↓
Vector Store
```

---

###  Code-Aware Processing

Instead of treating source code as ordinary text, the system is designed around code-specific structures such as:

* Files
* Classes
* Functions
* Methods
* Imports
* Modules

This allows retrieval to return meaningful pieces of the codebase.

---

###  Semantic Retrieval

User questions are converted into embeddings and compared against indexed code chunks.

For example:

```text
User Query
    ↓
"What handles authentication?"
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Relevant Code Chunks
```

This makes it possible to find relevant code even when the user's wording does not exactly match the source code.

---

###  LLM-Powered Answers

Retrieved code is supplied to the language model as context.

The LLM then generates an explanation based on the retrieved repository information.

```text
Question
    ↓
Retriever
    ↓
Relevant Code
    ↓
LLM
    ↓
Context-Aware Answer
```

---

###  Interactive UI

The project includes a Streamlit-based interface where developers can interact with the Codebase RAG system.

---

##  System Architecture

```text
                    ┌─────────────────────┐
                    │    GitHub / Local   │
                    │     Repository      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Repository Ingestion│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   File Filtering    │
                    │   & Code Parsing    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Code Chunking     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Embedding Generation│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Vector Store     │
                    │      Qdrant         │
                    └──────────┬──────────┘
                               │
                     User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Query Embedding    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Semantic Retrieval │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        LLM          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + References │
                    └─────────────────────┘
```

---

##  Project Structure

```text
CodeBase_RAG/
│
├── backend/
│   └── API/backend services
│
├── embeddings/
│   └── Embedding generation
│
├── ingestion/
│   ├── Repository loading
│   ├── File scanning
│   ├── File filtering
│   └── Code processing
│
├── llm/
│   └── LLM interaction and generation
│
├── retrieval/
│   └── Semantic retrieval logic
│
├── vector_store/
│   └── Vector database operations
│
├── ui/
│   └── Streamlit interface
│
├── tests/
│   └── Test cases
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Technology      | Purpose                        |
| --------------- | ------------------------------ |
| **Python**      | Core programming language      |
| **FastAPI**     | Backend/API layer              |
| **Streamlit**   | Interactive frontend           |
| **Qdrant**      | Vector database                |
| **Tree-sitter** | Code parsing                   |
| **NetworkX**    | Dependency/relationship graph  |
| **OpenAI**      | LLM capabilities               |
| **RAG**         | Retrieval-Augmented Generation |

The repository currently defines Python `>=3.11` in `pyproject.toml`.

---

##  Example Questions

Once a repository has been indexed, you can ask questions such as:

### Architecture

```text
How is this project structured?
```

### Authentication

```text
Where is authentication implemented?
```

### Database

```text
How does the application connect to the database?
```

### API Flow

```text
Explain what happens when a request reaches the API.
```

### Specific Function

```text
What does the RepositoryScanner class do?
```

### Dependencies

```text
Which modules depend on the retrieval system?
```

### Debugging

```text
Where could this error originate from?
```

---

##  RAG Pipeline

The application follows a Retrieval-Augmented Generation workflow.

### Step 1 — Repository Ingestion

The repository is scanned recursively to identify relevant source files.

Unnecessary files such as generated files, caches, and other ignored content can be excluded.

### Step 2 — Code Processing

Source files are processed and divided into meaningful chunks.

### Step 3 — Embedding Generation

Each chunk is converted into a numerical vector representation.

### Step 4 — Vector Storage

Embeddings and associated metadata are stored in the vector database.

### Step 5 — Query Processing

When a developer asks a question, the query is converted into an embedding.

### Step 6 — Retrieval

The system searches for code chunks that are semantically similar to the query.

### Step 7 — Context Construction

The retrieved code is assembled into context for the language model.

### Step 8 — Answer Generation

The LLM generates a response using the retrieved repository context.

---

##  Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Semantic search
* Vector databases
* Embeddings
* LLM application development
* Code parsing
* Repository indexing
* Information retrieval
* Backend API development
* Dependency graphs
* AI-assisted software engineering

---

##  Use Cases

###  Developers

Quickly understand unfamiliar codebases.

###  Software Engineers

Reduce the time required to navigate large repositories.

###  Students

Learn how real-world software projects are structured.

###  Code Review

Find relevant implementations and dependencies.

###  Debugging

Locate potentially relevant components for a reported issue.


---

##  Current Status

 **Under active development**

The current implementation focuses on building the core Codebase RAG pipeline, including repository ingestion, code processing, retrieval, vector storage, LLM integration, and the Streamlit interface.

---

##  Why This Project?

Codebase RAG explores how **RAG systems can move beyond document question-answering and become useful developer tools**.

The goal is to build an AI Software Engineer capable of understanding the structure and implementation of a real software repository and helping developers navigate it efficiently.

---

# CodeLens
CodeLens is an AI-powered codebase assistant that helps developers understand, explore, and interact with code repositories. It uses code ingestion, semantic embeddings, and Qdrant vector search to retrieve relevant code context, then leverages LLMs such as Google Gemini to generate intelligent answers. 
 07ac403185a209a693d26c4f71d8a9caf03ebf03
