#!/usr/bin/env python3
"""
Working RAG Application Example using YAVA Text Processing
==========================================================

This example demonstrates YAVA's current production capabilities for building
a RAG (Retrieval-Augmented Generation) application with text processing.

Current YAVA Pipeline: text → chunk → embed

Features demonstrated:
- Text document ingestion and processing
- Intelligent chunking with boundary preservation
- Vector embedding generation with LiteLLM integration
- Semantic similarity search with pgvector
- Context retrieval for RAG applications

Requirements:
- PostgreSQL with pgvector extension
- YAVA with production processors

Installation:
    uv sync

Usage:
    uv run python doc/examples/rag_text.py
    
    # Optional: Clean up demo data
    uv run python doc/examples/rag_text.py --cleanup
"""

import asyncio
import sys
from typing import List, Dict, Any

# YAVA imports - production ready components
from yava import Yava, DocPipeline, Document
from yava.processors import chunk_document, embed_chunk

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database configuration - adjust as needed
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres', 
    'password': 'postgres',
    'database': 'postgres'
}

# YAVA processing configuration
PROCESSING_CONFIG = {
    'chunking': {
        'chunk_size': 800,              # Target chunk size in characters
        'overlap_size': 80,             # Overlap between chunks for context
        'min_chunk_size': 100,          # Minimum viable chunk size
        'preserve_paragraphs': True     # Respect paragraph boundaries
    },
    'embeddings': {
        'model': 'vertex_ai/text-embedding-005',   # Requires real API key
        'retry_attempts': 3,                 # Retry failed embedding requests
        'normalize_vectors': True            # Normalize embedding vectors
    }
}

# Sample knowledge base documents for RAG demonstration
KNOWLEDGE_BASE = [
    {
        "title": "Introduction to Machine Learning",
        "content": """
        Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and models that enable computers to learn and make decisions from data without being explicitly programmed for every task. It represents a paradigm shift from traditional programming where rules are explicitly coded, to a system where patterns are learned from examples.

        The core principle behind machine learning is pattern recognition. By feeding large amounts of data to algorithms, these systems can identify relationships, trends, and patterns that might not be immediately obvious to humans. This capability makes machine learning particularly powerful for tasks involving complex data analysis, prediction, and decision-making.

        There are three main categories of machine learning approaches. Supervised learning uses labeled training data to teach algorithms to make predictions or classifications on new, unseen data. Common examples include email spam detection, image recognition, and medical diagnosis. Unsupervised learning finds hidden patterns in data without labeled examples, such as customer segmentation or anomaly detection. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors, commonly used in game playing, robotics, and autonomous systems.

        The applications of machine learning are vast and growing. In healthcare, ML algorithms help diagnose diseases, discover new drugs, and personalize treatment plans. In finance, they detect fraudulent transactions, assess credit risk, and enable algorithmic trading. In technology, ML powers recommendation systems, natural language processing, computer vision, and autonomous vehicles. The field continues to evolve rapidly with advances in deep learning, neural networks, and computational power.
        """,
        "metadata": {
            "category": "machine_learning",
            "difficulty": "beginner",
            "source": "educational_content",
            "topics": ["AI", "algorithms", "supervised_learning", "unsupervised_learning", "applications"]
        }
    },
    {
        "title": "Neural Networks and Deep Learning",
        "content": """
        Neural networks are computing systems inspired by the biological neural networks found in animal brains. They consist of interconnected nodes, called artificial neurons or units, organized in layers that process information through weighted connections. This architecture enables neural networks to learn complex patterns and relationships in data.

        A typical neural network consists of three types of layers: an input layer that receives data, one or more hidden layers that process information, and an output layer that produces results. Each connection between neurons has an associated weight that determines the strength and influence of the signal being transmitted. During training, these weights are adjusted using algorithms like backpropagation to minimize the difference between predicted and actual outputs.

        Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers (typically three or more) to learn hierarchical representations of data. The term "deep" refers to the number of layers in the network. Deep networks can automatically learn to identify relevant features at different levels of abstraction, from simple edges and shapes in images to complex concepts and relationships.

        Popular deep learning architectures include Convolutional Neural Networks (CNNs) for image processing and computer vision, Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sequential data and natural language processing, and Transformer networks that have revolutionized natural language understanding and generation. These architectures have enabled breakthroughs in image recognition, speech processing, machine translation, and artificial intelligence assistants.

        The training process involves feeding the network large amounts of data and adjusting weights through gradient descent optimization. This requires significant computational resources, typically using Graphics Processing Units (GPUs) or specialized hardware like Tensor Processing Units (TPUs). The success of deep learning has been driven by the availability of large datasets, increased computational power, and improved algorithms and architectures.
        """,
        "metadata": {
            "category": "deep_learning", 
            "difficulty": "intermediate",
            "source": "technical_documentation",
            "topics": ["neural_networks", "backpropagation", "CNN", "RNN", "transformers", "training"]
        }
    },
    {
        "title": "Vector Databases and Semantic Search",
        "content": """
        Vector databases are specialized data storage systems designed to store, index, and query high-dimensional vector representations of data such as text, images, audio, and other complex data types. Unlike traditional databases that store structured data in rows and columns, vector databases store mathematical representations (embeddings) that capture the semantic meaning and relationships of the original data.

        The key advantage of vector databases lies in their ability to perform semantic search rather than just keyword-based search. When you search for "car" in a traditional database, it only finds exact matches for that word. In a vector database, a search for "car" might also return relevant results for "automobile," "vehicle," "sedan," or "SUV" because these concepts are semantically similar and their vector representations are close together in the high-dimensional space.

        Vector similarity is typically measured using mathematical distance metrics such as cosine similarity, which measures the angle between vectors; Euclidean distance, which measures the straight-line distance between points; and dot product, which considers both magnitude and direction. The choice of similarity metric depends on the specific use case and the characteristics of the data being analyzed.

        To efficiently search through millions or billions of vectors, vector databases employ sophisticated indexing techniques. Hierarchical Navigable Small World (HNSW) graphs create a multi-layer structure that enables fast approximate nearest neighbor search. Inverted File (IVF) systems partition vectors into clusters for faster searching. Product Quantization compresses vectors while maintaining search accuracy. These techniques enable sub-second search across massive datasets.

        Applications of vector databases span many domains. In recommendation systems, they help find similar products or content based on user behavior and preferences. In semantic search engines, they enable natural language queries that understand intent rather than just matching keywords. In retrieval-augmented generation (RAG) systems, they provide relevant context to language models to improve response accuracy. Other applications include similarity detection, content moderation, anomaly detection, and personalization engines.
        """,
        "metadata": {
            "category": "vector_databases",
            "difficulty": "intermediate", 
            "source": "technical_guide",
            "topics": ["embeddings", "semantic_search", "similarity_metrics", "indexing", "HNSW", "applications"]
        }
    },
    {
        "title": "Natural Language Processing Fundamentals",
        "content": """
        Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language in a valuable way. It combines computational linguistics with machine learning and deep learning to bridge the gap between human communication and computer understanding.

        The field of NLP encompasses several core tasks and challenges. Text preprocessing involves cleaning and normalizing text data, including tokenization (splitting text into words or subwords), stemming and lemmatization (reducing words to their root forms), and handling punctuation and special characters. Named Entity Recognition (NER) identifies and classifies entities like people, organizations, locations, and dates within text. Part-of-speech tagging assigns grammatical categories to words, while syntactic parsing analyzes sentence structure and relationships between words.

        Modern NLP heavily relies on word embeddings and language models. Word embeddings like Word2Vec, GloVe, and FastText represent words as dense vectors that capture semantic relationships. More advanced contextual embeddings from models like BERT, GPT, and T5 consider the surrounding context to generate different representations for the same word in different contexts. These embeddings serve as the foundation for many downstream NLP tasks.

        Recent advances in transformer-based language models have revolutionized the field. Models like GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers), and their variants have achieved remarkable performance on a wide range of NLP tasks including text classification, question answering, summarization, translation, and text generation. The attention mechanism in transformers allows models to focus on relevant parts of the input when processing each word.

        NLP applications are ubiquitous in modern technology. Search engines use NLP to understand user queries and rank relevant documents. Virtual assistants like Siri and Alexa rely on NLP for speech recognition and natural language understanding. Machine translation services enable communication across language barriers. Sentiment analysis helps businesses understand customer feedback and social media opinions. Content generation tools assist with writing, while chatbots provide automated customer support.
        """,
        "metadata": {
            "category": "nlp",
            "difficulty": "intermediate",
            "source": "academic_content", 
            "topics": ["text_processing", "embeddings", "transformers", "BERT", "GPT", "applications"]
        }
    }
]

# Test questions for RAG demonstration
TEST_QUESTIONS = [
    "What is machine learning and how does it work?",
    "Explain the difference between supervised and unsupervised learning",
    "How do neural networks learn through backpropagation?", 
    "What are the advantages of vector databases over traditional databases?",
    "How do transformer models work in natural language processing?",
    "What is semantic search and how is it different from keyword search?",
    "Describe the applications of deep learning in real-world scenarios",
    "How do embeddings capture semantic meaning in NLP?"
]


async def main():
    """
    Complete YAVA RAG demonstration using functional approach.
    
    This function runs the entire RAG pipeline:
    1. Initialize YAVA with production processors
    2. Ingest knowledge base documents  
    3. Validate processing results
    4. Run question answering demonstration
    5. Optionally clean up demo data
    """
    print("🚀 YAVA RAG Text Processing Demo")
    print("=" * 60)
    print("Demonstrating production text → chunk → embed pipeline")
    print("=" * 60)
    
    # =============================================================================
    # STEP 1: Initialize YAVA
    # =============================================================================
    print("🚀 Initializing YAVA RAG System...")
    
    # Create text processing pipeline using production functions
    pipeline = DocPipeline("rag_text_pipeline") \
        .add_step("text", "chunk", chunk_document) \
        .add_step("chunk", "embed", embed_chunk)
    
    print("📋 Pipeline created: text → chunk → embed")
    
    # Initialize YAVA with database and pipeline
    yava = await Yava(
        **DB_CONFIG,
        document_types=[pipeline],
        process=True,  # Enable automatic processing
        config=PROCESSING_CONFIG
    )
    
    print("✅ YAVA initialized with production processors")
    
    try:
        # =============================================================================
        # STEP 2: Ingest Knowledge Base
        # =============================================================================
        print(f"\n📚 Ingesting {len(KNOWLEDGE_BASE)} documents into knowledge base...")
        
        document_ids = []
        for i, doc_data in enumerate(KNOWLEDGE_BASE, 1):
            print(f"  📄 Processing document {i}: {doc_data['title']}")
            
            # Create document with text content - triggers automatic processing
            doc = await yava.add(Document(
                content=doc_data['content'].strip(),
                state="text",  # Start with text state for processing
                metadata={
                    **doc_data['metadata'], 
                    'title': doc_data['title'],
                    'document_index': i,
                    'ingestion_time': str(asyncio.get_event_loop().time())
                }
            ))
            
            document_ids.append(doc.id)
            print(f"    ✅ Document ID: {doc.id}")
        
        print(f"\n⏳ Processing {len(document_ids)} documents through pipeline...")
        print("   🔄 text → chunk → embed")
        
        # Allow time for background processing to complete
        await asyncio.sleep(10)
        
        # =============================================================================
        # STEP 3: Validate Processing
        # =============================================================================
        print(f"\n🔍 Validating processing results...")
        
        stats = {
            'total_documents': len(document_ids),
            'total_chunks': 0,
            'embedded_chunks': 0,
            'failed_documents': 0
        }
        
        for doc_id in document_ids:
            try:
                # Get all chunks for this document
                chunks = await yava.list_documents(
                    filters={'parent_id': doc_id, 'state': 'chunk'}
                )
                embedded_chunks = await yava.list_documents(
                    filters={'parent_id': doc_id, 'state': 'embed'}
                )
                
                chunk_count = len(chunks)
                embedded_count = len(embedded_chunks)
                
                stats['total_chunks'] += chunk_count
                stats['embedded_chunks'] += embedded_count
                
                print(f"  📊 Document {doc_id[:8]}... → {chunk_count} chunks, {embedded_count} embedded")
                
                if embedded_count == 0:
                    stats['failed_documents'] += 1
                    
            except Exception as e:
                print(f"  ⚠️ Error validating document {doc_id}: {e}")
                stats['failed_documents'] += 1
        
        success_rate = ((stats['total_documents'] - stats['failed_documents']) / stats['total_documents'] * 100) if stats['total_documents'] > 0 else 0
        
        print(f"\n📈 Processing Results:")
        print(f"   • Documents processed: {stats['total_documents']}")
        print(f"   • Total chunks created: {stats['total_chunks']}")
        print(f"   • Embedded chunks: {stats['embedded_chunks']}")
        print(f"   • Success rate: {success_rate:.1f}%")
        
        if stats['embedded_chunks'] == 0:
            print("❌ No embedded chunks found - cannot proceed with search demo")
            print("💡 This may be due to:")
            print("   • Processing still in progress (try waiting longer)")
            print("   • Database connection issues")
            print("   • Embedding generation failures")
            return
        
        # =============================================================================
        # STEP 4: RAG Question Answering Demo
        # =============================================================================
        print("🎯 Running RAG Question Answering Demo")
        print("=" * 60)
        
        for i, question in enumerate(TEST_QUESTIONS[:5], 1):  # Run first 5 questions
            print(f"\n❓ Question {i}: {question}")
            print("-" * 60)
            
            # Search for relevant context
            try:
                context_results = await yava.search(
                    query=question,
                    limit=3,
                    similarity_threshold=0.5
                )
                
                if not context_results:
                    print("❌ No relevant context found in knowledge base.")
                    continue
                
                print(f"🎯 Found {len(context_results)} relevant context chunks:")
                
                # Display context
                for j, result in enumerate(context_results, 1):
                    similarity = result.similarity
                    content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
                    title = result.metadata.get('title', 'Unknown Document')
                    
                    print(f"\n  📋 Context {j} (similarity: {similarity:.3f}):")
                    print(f"     📖 Source: {title}")
                    print(f"     📝 Content: {content_preview}")
                
                # Combine context for potential LLM input
                combined_context = "\n\n".join([result.content for result in context_results])
                avg_similarity = sum([result.similarity for result in context_results]) / len(context_results)
                
                print(f"\n💡 RAG Context Summary:")
                print(f"   • Total context length: {len(combined_context)} characters")
                print(f"   • Average similarity: {avg_similarity:.3f}")
                print(f"   • Sources: {len(set([r.metadata.get('title', 'Unknown') for r in context_results]))}")
                print(f"   • Ready for LLM processing (context + question → answer)")
                
            except Exception as e:
                print(f"⚠️ Search error: {e}")
            
            if i < 5:
                print("\n" + "="*60)
        
        # =============================================================================
        # STEP 5: Demo Summary
        # =============================================================================
        print(f"\n" + "="*60)
        print("📊 Demo Summary:")
        print(f"   • Questions processed: 5")
        print(f"   • Documents in knowledge base: {len(document_ids)}")
        print(f"   • Total chunks available for search: {stats['embedded_chunks']}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Optional cleanup
        if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
            print("\n🧹 Cleaning up demo data...")
            
            deleted_count = 0
            try:
                # Delete all documents (cascades to chunks and embeddings)
                for doc_id in document_ids:
                    await yava.delete(doc_id)
                    deleted_count += 1
                
                print(f"✅ Cleaned up {deleted_count} documents and related data")
                
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")


async def cleanup_only():
    """Clean up demo data only."""
    print("🧹 Cleanup mode - removing all demo data...")
    
    # Initialize YAVA for cleanup
    yava = await Yava(**DB_CONFIG)
    
    try:
        # Get all documents to clean up
        all_docs = await yava.list_documents(limit=1000)
        
        if not all_docs:
            print("✅ No documents found to clean up")
            return
        
        print(f"🗑️ Deleting {len(all_docs)} documents...")
        
        deleted_count = 0
        for doc in all_docs:
            await yava.delete(doc.id)
            deleted_count += 1
        
        print(f"✅ Cleaned up {deleted_count} documents and related data")
        
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        asyncio.run(cleanup_only())
    else:
        asyncio.run(main())