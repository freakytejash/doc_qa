#!/usr/bin/env python3
"""
Quick demo script to showcase RAG system capabilities
Run this to verify everything works before interview
"""

from core.embeddings import get_embedder
from core.vectorstore import query_collection
from core.llm import get_llm

def demo():
    print("=" * 60)
    print("📚 RAG SYSTEM DEMO")
    print("=" * 60)
    
    # Demo query
    question = "Based on page 515, describe the parallel parlor design"
    
    print(f"\n❓ Question: {question}\n")
    
    # Retrieve relevant chunks
    embedder = get_embedder()
    chunks = query_collection(
        query_embedding=embedder.embed_one(question), 
        top_k=5
    )
    
    print("📄 Retrieved Chunks (with relevance scores):")
    print("-" * 60)
    for i, chunk in enumerate(chunks, 1):
        meta = chunk['metadata']
        page_info = f"PDF page {meta.get('page')}"
        if meta.get('textbook_page'):
            page_info += f" (textbook p.{meta.get('textbook_page')})"
        
        score = round(1 - chunk['distance'], 3)
        print(f"\n[{i}] {page_info} | Relevance: {score}")
        print(f"    {chunk['text'][:150]}...")
    
    # Generate answer
    print("\n" + "=" * 60)
    print("🤖 LLM Answer:")
    print("-" * 60)
    llm = get_llm()
    answer = llm.answer(question, chunks)
    print(answer)
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Upload PDF via: streamlit run streamlit_app.py")
    print("2. Try different questions to see how retrieval adapts")
    print("3. Watch how page metadata improves answer grounding")

if __name__ == "__main__":
    demo()
