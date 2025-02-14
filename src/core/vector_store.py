from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import json
import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class OpenAIEmbeddingFunction:
    """Custom embedding function for ChromaDB"""
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if isinstance(input, str):
            input = [input]
        return [self.embeddings.embed_query(text) for text in input]

class VectorStore:
    def __init__(self, model: str = "text-embedding-3-small", persist_directory: str = "data/chroma_db"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model=model
        )
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

    def create_vector_store(self, embeddings_file: str):
        """Create a collection with pre-computed embeddings."""
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        with open(embeddings_file, 'r') as f:
            data = json.load(f)['embeddings']

        # Create or get collection
        if not self.collection:
            self.collection = self.client.get_or_create_collection(
                name="academic_data",
                embedding_function=self.embedding_function
            )

        # Get existing document IDs to prevent duplicate insertions
        existing_ids = set(self.collection.get()['ids'])

        # Prepare new data, avoiding duplicates
        new_embeddings = []
        new_documents = []
        new_metadatas = []
        new_ids = []

        for idx, item in enumerate(data):
            doc_id = f"doc_{idx}"
            if doc_id not in existing_ids:  # Only add new documents
                new_embeddings.append(item['embedding'])
                new_documents.append(item['text'])
                new_metadatas.append(item['metadata'])
                new_ids.append(doc_id)

        # Add new data to collection
        if new_embeddings:
            self.collection.add(
                embeddings=new_embeddings,
                documents=new_documents,
                metadatas=new_metadatas,
                ids=new_ids
            )
            print(f"\nAdded {len(new_embeddings)} new documents to the vector store.")
        else:
            print("\nNo new documents were added. All embeddings already exist.")

        self._print_store_summary(new_metadatas)
        return self.collection

    def similarity_search(self, query: str, filter_dict: Dict = None, k: int = 16):
        """Search using vector similarity."""
        if not self.collection:
            raise ValueError("Collection not initialized.")

        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query])[0] # Get first embedding since we only have one query
            

            print(f"\nPerforming similarity search for query: {query}")
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas']
            )

            # Format results
            formatted_results = [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(results['documents'][0], results['metadatas'][0])
            ]
            print(f"Search results found: {len(formatted_results)}")
            return formatted_results

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

   
    def _print_store_summary(self, metadatas: List[Dict]):
        """Print summary of vector store contents."""
        types = [m['type'] for m in metadatas]
        print(f"\nVector store created with {len(types)} documents:")
        print(f"- Teachers: {types.count('teacher')}")
        print(f"- Classes: {types.count('class')}")
        print(f"- Students: {types.count('student')}")

    def cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(vector_a)
        b = np.array(vector_b)
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def cosine_similarity_search(
        self, 
        query: str, 
        filter_dict: Optional[Dict] = None, 
        k: int = 16,
        score_threshold: Optional[float] = 0.3
    ) -> List[Dict]:
        """
        Search using cosine similarity with optional filtering and score thresholding.
        
        Args:
            query: Search query string
            filter_dict: Optional dictionary of metadata filters
            k: Number of results to return
            score_threshold: Minimum similarity score threshold (0 to 1)
            
        Returns:
            List of dictionaries containing content, metadata, and similarity scores
        """
        if not self.collection:
            raise ValueError("Collection not initialized.")

        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query])[0]
            
            print(f"\nPerforming cosine similarity search for query: {query}")
            
            # Get all documents and their embeddings
            collection_data = self.collection.get(
                include=['embeddings', 'documents', 'metadatas']
            )
            
            # Calculate similarities and create result objects
            similarities = []
            for idx, (doc_embedding, doc, metadata) in enumerate(zip(
                collection_data['embeddings'],
                collection_data['documents'],
                collection_data['metadatas']
            )):
                # Apply metadata filtering if specified
                if filter_dict and not all(
                    metadata.get(k) == v for k, v in filter_dict.items()
                ):
                    continue
                
                # Calculate cosine similarity
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                
                # Apply score threshold
                if similarity >= score_threshold:
                    similarities.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'index': idx
                    })
            
            # Sort by similarity score in descending order
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Take top k results
            results = similarities[:k]
            
            # Print detailed search statistics
            print(f"\nSearch statistics:")
            print(f"- Total documents processed: {len(similarities)}")
            print(f"- Results above threshold {score_threshold}: {len(results)}")
            if not results:
                # If no results with current threshold, find the highest similarity score
                if similarities:
                    max_similarity = max(similarities, key=lambda x: x['similarity'])
                    print(f"- Highest similarity score found: {max_similarity['similarity']:.3f}")
                    print("- Consider lowering the score_threshold to see more results")
                
            return results

        except Exception as e:
            print(f"Cosine similarity search error: {str(e)}")
            return []

def main():
    """Test vector store functionality."""
    try:
        # Initialize vector store
        print("\nInitializing vector store...")
        vector_store = VectorStore()

        # Create vector store from embeddings
        print("\nCreating vector store from embeddings...")
        embeddings_file = "data/embeddings.json"
        vector_store.create_vector_store(embeddings_file)

        # Test queries
        test_queries = [
            "Who teaches mathematics?",
            "List all students in Class 3A",
            "What are the features of Class 3B?",
            "Tell me about the art teacher"
        ]

        print("\nTesting both search queries...")
        for query in test_queries:
            print("\nOriginal Similarity Search Results:")
            results = vector_store.similarity_search(query, k=5)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Content: {result['content']}")
                print(f"   Metadata: {result['metadata']}")

             # Test cosine similarity search
            print("\nCosine Similarity Search Results:")
            cosine_results = vector_store.cosine_similarity_search(
                query=query,
                k=5,
                score_threshold=0.3
            )
            for i, result in enumerate(cosine_results, 1):
                print(f"\n{i}. Content: {result['content']}")
                print(f"   Metadata: {result['metadata']}")
                print(f"   Similarity Score: {result['similarity']:.3f}")


    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()

