from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self, model: str = "text-embedding-3-small", persist_directory: str = "data/chroma_db"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=openai_api_key
        )
        self.persist_directory = persist_directory # Where vector data is stored
        self.chroma_store = None # ChromaDB instance

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

    
    def create_vector_store(self, embeddings_file: str):
        """Create a Chroma vector store from embedding data."""
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
            if 'embeddings' not in data:
                raise ValueError("Invalid embeddings file format")
            embeddings_data = data['embeddings']

        texts, embeddings, metadatas = [], [], []
        for item in embeddings_data:
            texts.append(item['text'])
            embeddings.append(item['embedding'])
            metadatas.append({
                'type': self._get_content_type(item['text']),
                'class': self._extract_class_reference(item['text'])
            })

        self.chroma_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_directory
        )

        self._print_store_summary(metadatas)
        return self.chroma_store

    def _get_content_type(self, text: str) -> str:
        """Get content type from text"""
        if text.startswith("Teacher Information:"):
            return "teacher"
        elif text.startswith("Class Information:"):
            return "class"
        elif text.startswith("Student Information:"):
            return "student"
        return "unknown"

    def _extract_class_reference(self, text: str) -> str:
        """Extract class reference from text"""
        for class_name in ["3A", "3B", "3C"]:
            if f"Class {class_name}" in text:
                return class_name
        return "unknown"

    def _print_store_summary(self, metadatas: List[Dict]):
        """Print summary of vector store contents."""
        types = [m['type'] for m in metadatas]
        print(f"\nVector store created with {len(types)} documents:")
        print(f"- Teachers: {types.count('teacher')}")
        print(f"- Classes: {types.count('class')}")
        print(f"- Students: {types.count('student')}")

    def similarity_search(self, query: str, filter_dict: Dict = None, k: int = 20):
        """Perform similarity search with metadata filtering.
    
        Args:
            query (str): The search query
            filter_dict (Dict, optional): Metadata filters
            k (int, optional): Number of results to return. Defaults to 7.
        """

        if not self.chroma_store:
            raise ValueError("Vector store not initialized.")
        
        # Enhanced query type detection
        content_type = self._get_query_type(query)
        class_name = self._get_class_from_query(query)
        
        try:
            # Construct the filter using ChromaDB's specific format
            if content_type and class_name:
                # Use AND condition for multiple filters
                filter_dict = {
                    "$and": [
                        {"type": {"$eq": content_type}},
                        {"class": {"$eq": class_name}}
                    ]
                }
            elif content_type:
                filter_dict = {"type": {"$eq": content_type}}
            elif class_name:
                filter_dict = {"class": {"$eq": class_name}}
            else:
                filter_dict = None

            # First, try a filtered search
            if filter_dict:
                try:
                    results = self.chroma_store.similarity_search(
                        query,
                        filter=filter_dict,
                        k=k
                    )
                    if results:
                        return self._get_unique_results(results, k=k)
                except Exception as e:
                    print(f"Filtered search error: {str(e)}")
            
            # Fallback to unfiltered search if filtered search fails or returns no results
            results = self.chroma_store.similarity_search(query, k=k)
            return self._get_unique_results(results, k=k)

        except Exception as e:
            print(f"Search error: {str(e)}")
            # Fallback to unfiltered search
            return self.chroma_store.similarity_search(query, k=k)


    def _get_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query = query.lower()
        if "student" in query or "students" in query: return "student"
        if "teacher" in query or "teachers" in query: return "teacher"
        if "class" in query or "classes" in query: return "class"
        return None

    def _get_class_from_query(self, query: str) -> str:
        """Extract class reference from query."""
        query = query.upper()
        for class_name in ["3A", "3B", "3C"]:
            if class_name in query:
                return class_name
        return None

    def _get_unique_results(self, results: List, k: int = 7) -> List:
        """Remove duplicates while preserving order.
        
        Args:
            results (List): List of search results
            k (int): Maximum number of results to return
        """
        seen_content = set()
        unique_results = []
        for doc in results:
            # Use page content as the primary uniqueness key
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_results.append(doc)
                
                if len(unique_results) >= k:
                    break

        return unique_results

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

        # Test different types of queries
        test_queries = [
            "Who teaches mathematics?",
            "List all students in Class 3C.",
        ]

        print("\nTesting search queries...")
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            # Search with default parameters
            results = vector_store.similarity_search(query)
            print(f"Found {len(results)} results:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.page_content}")

            # Search with type filter
            if "class" in query.lower():
                filtered_results = vector_store.similarity_search(
                    query, 
                    filter_dict={"type": "class"},
                    k=5
                )
                print(f"\nFiltered results (class only):")
                for i, doc in enumerate(filtered_results, 1):
                    print(f"\n{i}. {doc.page_content}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
   main()