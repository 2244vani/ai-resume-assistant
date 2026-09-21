# ai-resume-assistant
AI-powered Resume Assistant that uses LangChain, LangGraph, RAG, ChromaDB, Ollama, and Llama 3.2 to analyze resumes against job descriptions, identify matching and missing skills, calculate match percentage, and provide improvement suggestions.

# 📄 AI Resume Assistant

An AI-powered Resume Assistant that analyzes a candidate's resume against a Job Description (JD). It identifies matching skills, missing skills, preferred skills, calculates a resume match percentage, and provides suggestions for improving the resume.

## 🚀 Features

- 📄 Upload resume in PDF format
- 🔍 Extract and process resume text
- ✂️ Split resume into smaller chunks
- 🧠 Generate embeddings using Ollama
- 🗄️ Store embeddings in ChromaDB
- 🔎 Retrieve relevant resume information using RAG
- 🤖 Analyze resume using Llama 3.2
- 📋 Analyze Job Description
- ✅ Identify matching required skills
- ❌ Identify missing required skills
- ⭐ Identify preferred skills
- 📊 Calculate resume match percentage
- 💡 Provide resume improvement suggestions
- 🔄 Use LangGraph for AI workflow
- 🖥️ Simple Streamlit web interface

## 🛠️ Technologies Used

- Python
- LangChain
- LangGraph
- Ollama
- Llama 3.2
- ChromaDB
- RAG
- Streamlit
- PyPDFLoader
- Nomic Embeddings

## 🏗️ Project Workflow

```text
Resume PDF
    ↓
PDF Loader
    ↓
Text Splitting
    ↓
Ollama Embeddings
    ↓
ChromaDB
    ↓
RAG Similarity Search
    ↓
LangGraph Workflow
    ↓
Llama 3.2
    ↓
Resume Analysis
    ↓
Match Percentage + Skills + Suggestions
