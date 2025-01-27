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
        self.persist_directory = persist_directory
        self.chroma_store = None

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
    
    def load_embeddings(self, embeddings_file: str) -> List[Dict]:
        """Load embedding data from a JSON file."""
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
            if 'embeddings' not in data:
                raise ValueError("Invalid embeddings file format: Missing 'embeddings' key.")
            return data['embeddings']
    
    def create_vector_store(self, embeddings_file: str):
        """Create a Chroma vector store from embedding data."""
        embeddings_data = self.load_embeddings(embeddings_file)
            
        # Debug print
        print(f"Loaded {len(embeddings_data)} documents")

        texts = []
        embeddings = []
        metadatas = []
            
        for item in embeddings_data:
            text = item['text']
            embedding = item['embedding']
                
            # Create metadata based on content
            metadata = {
                'type': self._get_content_type(text),
                'class': self._extract_class_reference(text)
            }
                
            texts.append(text)
            embeddings.append(embedding)
            metadatas.append(metadata)
            
            
        # Create vector store
        self.chroma_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_directory
        )
            
        # Print summary
        types = [m['type'] for m in metadatas]
        print(f"\nVector store created with {len(texts)} documents:")
        print(f"- Teachers: {types.count('teacher')}")
        print(f"- Classes: {types.count('class')}")
        print(f"- Students: {types.count('student')}")
        
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

    def similarity_search(self, query: str, filter_dict: Dict = None, k: int = 7):
        """Perform similarity search with metadata filtering."""
        if not self.chroma_store:
            raise ValueError("Vector store not initialized.")
            
        # Get the content type from the query
        content_type = None
        if "teacher" in query.lower():
            content_type = "teacher"
        elif "class" in query.lower():
            content_type = "class"
        elif "student" in query.lower():
            content_type = "student"
            
        # Add type filter if detected
        if content_type and not filter_dict:
            filter_dict = {"type": content_type}
            
        try:
            # Use search_kwargs for k parameter
            results = self.chroma_store.similarity_search(
                query,
                filter=filter_dict,
            )
        
            # Remove duplicates while preserving order
            seen = set()
            unique_results = []
            for doc in results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_results.append(doc)
                    
            return unique_results
        
        except Exception as e:
            print(f"Search error: {str(e)}")
            # Fallback to basic search without parameters
            return self.chroma_store.similarity_search(query)