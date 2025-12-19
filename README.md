# Multilingual Knowledge RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system built with Google's Gemini AI that processes multilingual PDF documents and provides intelligent question-answering capabilities.

##  Overview

This project implements an advanced RAG pipeline that:
- Extracts text and structure from PDF documents using OCR
- Processes multilingual content (supports Gujarati, Sanskrit, Hindi, English, and more)
- Creates semantic embeddings using state-of-the-art sentence transformers
- Builds a FAISS vector index for efficient similarity search
- Generates contextually accurate answers using Google's Gemini 1.5 Pro model

## Key Features

### 1. **Intelligent Document Processing**
- **PDF OCR**: Converts PDF pages to images and extracts text using Google's Gemini Vision API
- **Multilingual Support**: Handles multiple languages including Indic scripts
- **Structured Extraction**: Preserves document hierarchy (headings, paragraphs, page numbers)
- **Caching**: Saves OCR results to avoid redundant API calls

### 2. **Advanced Knowledge Base**
- **Hierarchical Chunking**: Intelligently splits content while preserving context
- **Semantic Embeddings**: Uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for multilingual embeddings
- **Vector Search**: FAISS index for fast similarity-based retrieval
- **Context Preservation**: Maintains document structure and metadata

### 3. **Intelligent Question Answering**
- **RAG Pipeline**: Retrieves relevant context before generating answers
- **Gemini 1.5 Pro**: Leverages Google's latest LLM for accurate responses
- **Multilingual Responses**: Answers in the same language as the source content
- **Source Attribution**: Provides page numbers and context for answers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG System Architecture                   │
└─────────────────────────────────────────────────────────────┘

1. Document Processing Layer
   ├── PDF Input
   ├── Image Conversion (pdf2image)
   ├── OCR (Gemini Vision API)
   └── Structured JSON Output

2. Knowledge Base Layer
   ├── Hierarchical Chunking
   ├── Embedding Generation (Multilingual MPNet)
   ├── FAISS Vector Index
   └── Metadata Storage

3. Retrieval Layer
   ├── Query Embedding
   ├── Similarity Search (FAISS)
   └── Context Ranking

4. Generation Layer
   ├── Context Assembly
   ├── Prompt Engineering
   ├── Gemini 1.5 Pro Generation
   └── Response Formatting
```

## Prerequisites

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended for large documents)
- Internet connection (for API calls)

### API Keys
- **Google AI Studio API Key**: Required for Gemini API access
  - Get your key from: https://makersuite.google.com/app/apikey

##  Installation

### 1. Clone or Download the Project
```bash
cd "c:\Ai project\Test2"
```

### 2. Install Dependencies
```bash
pip install google-generativeai
pip install sentence-transformers
pip install faiss-cpu
pip install pdf2image
pip install Pillow
pip install numpy
```

### 3. Install Poppler (Required for pdf2image)

**Windows:**
1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler`
3. Add `C:\Program Files\poppler\Library\bin` to your system PATH

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### 4. Configure API Key
Set your Gemini API key as an environment variable:

**Windows:**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or modify the notebook to include your API key directly:
```python
GEMINI_API_KEY = "your-api-key-here"
```

##  Usage

### Running the Jupyter Notebook

1. **Open the Notebook**
   ```bash
   jupyter notebook "Multilingual_Knowledge.ipynb"
   ```

2. **Configure Your PDF**
   Update the configuration in the notebook:
   ```python
   class Config:
       PDF_PATH = "path/to/your/document.pdf"
       OUTPUT_JSON = "structured_knowledge.json"
       GEMINI_API_KEY = "your-api-key"
   ```

3. **Run All Cells**
   - The system will process your PDF
   - Build the knowledge base
   - Start an interactive Q&A session

### Interactive Q&A
Once the system is ready, you can ask questions:

```
 GITA AI ASSISTANT READY
==================================================

Ask a question (or 'exit'): What is the main topic of this document?
Thinking...

Answer:
[AI-generated answer based on document content]
```

##  Configuration Options

### Config Class Parameters

```python
class Config:
    # Input/Output Paths
    PDF_PATH = "test.pdf"              # Path to your PDF file
    OUTPUT_JSON = "structured_knowledge.json"  # OCR cache file
    
    # API Configuration
    GEMINI_API_KEY = "your-key"        # Your Gemini API key
    
    # Model Settings
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    LLM_MODEL = "gemini-1.5-pro"
    
    # Chunking Parameters
    CHUNK_SIZE = 500                   # Characters per chunk
    CHUNK_OVERLAP = 100                # Overlap between chunks
    
    # Retrieval Settings
    TOP_K = 3                          # Number of chunks to retrieve
```

##  Project Structure

```
Test2/
├── Multilingual_Knowledge.ipynb    # Main Jupyter notebook
├── test.pdf                         # Sample PDF document
├── Rag.pdf                          # Documentation PDF
├── structured_knowledge.json        # OCR cache (generated)
├── README.md                        # This file
└── DOCUMENTATION.md                 # Detailed technical documentation
```

##  Core Components

### 1. DocumentProcessor Class
Handles PDF processing and OCR:
- Converts PDF pages to images
- Sends images to Gemini Vision API
- Extracts structured text with metadata
- Caches results to JSON

### 2. KnowledgeBase Class
Manages the vector database:
- Chunks documents hierarchically
- Generates embeddings
- Builds FAISS index
- Handles similarity search

### 3. RAGAssistant Class
Orchestrates the Q&A pipeline:
- Processes user queries
- Retrieves relevant context
- Generates answers using Gemini
- Formats responses

##  How It Works

### Step 1: Document Processing
```python
processor = DocumentProcessor(Config)
raw_data = processor.process_pdf()
```
- Converts each PDF page to an image
- Uses Gemini Vision to extract text
- Preserves document structure (headings, paragraphs)
- Saves to JSON for reuse

### Step 2: Knowledge Base Creation
```python
kb = KnowledgeBase(Config)
kb.chunk_data(raw_data)
kb.build_index()
```
- Splits text into semantic chunks
- Generates multilingual embeddings
- Creates FAISS vector index
- Stores metadata for each chunk

### Step 3: Question Answering
```python
bot = RAGAssistant(Config, kb)
answer = bot.ask("Your question here")
```
- Embeds the user's question
- Finds most similar chunks (Top-K retrieval)
- Constructs context-aware prompt
- Generates answer using Gemini 1.5 Pro

##  Multilingual Support

The system supports multiple languages including:
- **English**
- **Hindi** (हिंदी)
- **Gujarati** (ગુજરાતી)
- **Sanskrit** (संस्कृत)
- **And many more** (via multilingual embeddings)

The embedding model `paraphrase-multilingual-mpnet-base-v2` and `intfloat/multilingual-e5-large` supports 50+ languages.

##  Performance Considerations

### Optimization Tips

1. **Chunk Size**: Adjust based on your document type
   - Technical docs: 300-500 characters
   - Narrative text: 500-800 characters

2. **Top-K Retrieval**: Balance between context and noise
   - Simple questions: K=1-2
   - Complex questions: K=3-5

3. **Caching**: The system caches OCR results
   - First run: Slower (OCR processing)
   - Subsequent runs: Fast (loads from JSON)

4. **Memory**: FAISS index size depends on document length
   - Small docs (<50 pages): <100MB RAM
   - Large docs (>200 pages): 500MB+ RAM

##  Example Use Cases

### 1. Religious Text Analysis
```python
# Example: Bhagavad Gita analysis
question = "What is the significance of the Conch Sound?"
answer = bot.ask(question)
```

### 2. Technical Documentation
```python
# Example: API documentation
question = "How do I authenticate with the API?"
answer = bot.ask(question)
```

### 3. Research Papers
```python
# Example: Scientific paper
question = "What methodology was used in this study?"
answer = bot.ask(question)
```

##  Troubleshooting

### Common Issues

**1. "Poppler not found" Error**
```
Solution: Install Poppler and add to PATH (see Installation section)
```

**2. "API Key Invalid" Error**
```
Solution: Verify your Gemini API key is correct and active
```

**3. "Out of Memory" Error**
```
Solution: 
- Reduce CHUNK_SIZE
- Process fewer pages at once
- Use a machine with more RAM
```

**4. "Model Download Slow"**
```
Solution: The embedding model (2.2GB) downloads on first run.
Be patient or use a faster internet connection.
```

##  References

### Technologies Used
- **Google Gemini AI**: https://ai.google.dev/
- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **pdf2image**: https://github.com/Belval/pdf2image

### Inspired By
This project was inspired by the RAG implementation from:
- GitHub: https://github.com/noobie105/10MS_RAG_Application

