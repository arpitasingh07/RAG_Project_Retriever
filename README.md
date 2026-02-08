# Hybrid RAG System - Group 61

A comprehensive Hybrid Retrieval-Augmented Generation (RAG) system built for conversational AI with advanced evaluation capabilities.

## Overview

This system implements a state-of-the-art RAG pipeline that combines BM25 sparse retrieval with dense semantic retrieval, enhanced by Reciprocal Rank Fusion (RRF) and cross-encoder reranking. The system is designed for strict extractive question answering with robust hallucination resistance.

## Key Features

- **Hybrid Retrieval**: Combines BM25 and dense retrieval with RRF fusion
- **Cross-Encoder Reranking**: Uses MS-MARCO MiniLM for improved ranking
- **Strict Extractive QA**: Generates answers only from retrieved context
- **Hallucination Resistance**: Built-in fallback for unanswerable questions
- **Comprehensive Evaluation**: Automated metrics and adversarial testing
- **Interactive Dashboard**: Gradio-based evaluation interface

## System Architecture

```
Query → Hybrid Search (BM25 + Dense + RRF) → Cross-Encoder Reranking → Extractive Answer Generation → Output
```

## Installation

### Prerequisites

- Python 3.8 or higher
- GPU support (optional, for faster processing)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd convAI
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required data**:
   ```bash
   python -m nltk.downloader punkt
   ```

## Dependencies

- `wikipedia-api`: For Wikipedia content retrieval
- `sentence-transformers`: For dense embeddings
- `faiss`: For efficient vector similarity search
- `rank_bm25`: For BM25 retrieval
- `transformers`: For LLM-based answer generation
- `gradio`: For interactive dashboard
- `pandas`: For data processing
- `matplotlib`: For visualization
- `base64`: For embedding images in html report

## Data Preparation

### 1. Wikipedia Corpus Collection

The system uses a diverse Wikipedia corpus of 500 articles (200 fixed + 300 random topics):


### 2. Text Processing

- **Cleaning**: Removes extra whitespace and formatting
- **Chunking**: Splits text into 200-400 token chunks with 50-token overlap
- **Total Chunks**: 6,458 chunks stored in final_corpus.json

### 3. Embedding Generation

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS Index**: Built for efficient similarity search
- **Normalization**: Cosine similarity optimization


### 2. Interactive Dashboard

```python
import gradio as gr
import time

# Launch interactive interface
app = gr.Interface(
    fn=gradio_rag,
    inputs="text",
    outputs="markdown",
    title="Hybrid RAG System Group 61"
)

app.launch(debug=True)
```

### 2. Cross-Encoder Reranking

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for pairwise relevance scoring.

### 3. Adversarial Testing

Evaluates system robustness with:
- Negation questions
- Multi-hop reasoning
- Ambiguous pronouns
- Unanswerable queries
- Paraphrased questions

### 4. Automated Evaluation Pipeline

```python
# Run full evaluation with one command
python evaluate_pipeline.py

# Outputs: CSV and JSON reports with all metrics
```

## Error Analysis

### Common Error Types

1. **Retrieval Errors** : Wrong Wikipedia page retrieved
2. **Ambiguous Questions** : Pronoun resolution issues
3. **Reasoning Errors** : Multi-hop comparison failures
4. **Hallucinations** : Yes/no reasoning errors

### Error Handling

- **Fallback Mechanism**: "The answer is not available in the given context"
- **Groundedness Detection**: Prevents hallucination propagation
- **Context Validation**: Ensures answer faithfulness


## Performance Optimization

### GPU Acceleration

For faster processing:
```python
# Set device to GPU
import torch
torch.cuda.set_device(0)

# Use GPU-enabled models
model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
```

### Model Optimization

- **Quantization**: Reduce model size
- **Pruning**: Remove redundant parameters
- **Batching**: Process multiple queries


## Troubleshooting

### Common Issues

1. **Wikipedia API Rate Limiting**:
   - Add delays between requests
   - Use caching
   - Handle connection errors

2. **Memory Issues**:
   - Reduce batch size
   - Use FAISS with smaller index
   - Enable GPU if available

3. **Model Loading Errors**:
   - Check internet connection
   - Verify model names
   - Ensure sufficient disk space

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Improvements

1. **Entity Linking**: Better entity resolution
2. **Context Expansion**: Retrieve more relevant passages
3. **Reasoning Models**: Add multi-hop reasoning capabilities
4. **Real-time Updates**: Dynamic corpus updates
5. **Multi-language Support**: Expand beyond English



## License

This project is licensed under the MIT License - see the LICENSE file for details.
