# RAG Image Search with CLIP & Gemini

A production-ready Retrieval-Augmented Generation (RAG) system for semantic image search using OpenAI's CLIP model and Google's Gemini for intelligent ranking. This system enables natural language queries to find relevant images from the Unsplash dataset with high accuracy and low latency.

## System Architecture

### High-Level Architecture
```
User Query → CLIP Text Encoder → Vector Similarity Search → Top-K Candidates → Gemini Ranking → Final Results
```

### Data Flow Architecture
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│ User Query  │ →  │ CLIP Encoder │ →  │ Vector DB   │ →  │ Top 10     │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐       │
│ Final 4     │ ←  │ Gemini      │ ←  │ Image       │ ←─────┘
│ Images      │    │ Ranking     │    │ Analysis    │
└─────────────┘    └──────────────┘    └─────────────┘
```

