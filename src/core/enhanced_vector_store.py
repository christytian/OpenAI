from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
import chromadb
import json
import os
import shutil
from typing import List, Dict, Optional
from dotenv import load_dotenv

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
        
        # Clean up existing directory if it exists
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Cleaned up existing directory: {persist_directory}")
        
        # Create fresh client and collection
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Enhanced metadata fields with email information
        self.metadata_field_info = [
            AttributeInfo(
                name="type",
                description="The type of record (teacher, student, or class)",
                type="string",
            ),
            AttributeInfo(
                name="class",
                description="The class identifier (3A, 3B, or 3C)",
                type="string",
            ),
            AttributeInfo(
                name="email",
                description="Email address for contact",
                type="string",
            ),
            AttributeInfo(
                name="parent_email",
                description="Parent's email address for students",
                type="string",
            ),
            AttributeInfo(
                name="head_teacher_email",
                description="Head teacher's email address for classes",
                type="string",
            ),
            AttributeInfo(
                name="subject",
                description="Subject taught by teacher",
                type="string",
            )
        ]

    def create_vector_store(self, embeddings_file: str):
        """Create a collection with pre-computed embeddings."""
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        # Load embeddings with enhanced metadata
        with open(embeddings_file, 'r') as f:
            data = json.load(f)['embeddings']
        
        # Create new collection
        self.collection = self.client.create_collection(
            name="embeddings_store",
            embedding_function=self.embedding_function
        )
        print("\nCreated new collection: embeddings_store")

        # Prepare data
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for idx, item in enumerate(data):
            documents.append(item['text'])
            embeddings.append(item['embedding'])
            metadatas.append(item['metadata'])
            ids.append(f"doc_{idx}")

        # Add documents
        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} documents to the vector store.")

        # Create Langchain wrapper
        self.langchain_vectorstore = Chroma(
            client=self.client,
            collection_name="embeddings_store",
            embedding_function=self.embedding_function.embeddings
        )

        # Initialize retrievers
        self._initialize_retrievers()
        
        # Print summary
        self._print_store_summary(metadatas)
        return self.langchain_vectorstore

    def _initialize_retrievers(self):
        """Initialize retrievers with enhanced document description."""
        document_content_description = """
        Academic database containing:
        1. Teacher records: Names, subjects, class assignments, email addresses, and office hours
        2. Student records: Names, ages, class assignments, student emails, and parent contact information
        3. Class records: Locations, features, head teachers, head teacher emails, and capacity
        
        Contact information is available for all records including:
        - Teacher email addresses and office hours
        - Student email addresses and parent contact information
        - Head teacher contact information for each class
        """

        self.retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.langchain_vectorstore,
            document_contents=document_content_description,
            metadata_field_info=self.metadata_field_info,
            verbose=False
        )
        
        # Enhanced context compressor with email awareness
        email_aware_prompt = PromptTemplate.from_template(
            """Given the question and context, provide the directly relevant information, 
            ensuring to include any relevant contact information or email addresses.

            Question: {question}
            Context: {context}
            
            Relevant Information:"""
        )
        
        compressor = LLMChainExtractor.from_llm(
            self.llm,
            prompt=email_aware_prompt
        )
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )

    def similarity_search(
        self, 
        query: str, 
        filter_dict: Optional[Dict] = None, 
        k: int = 5
    ) -> List[Document]:
        """
        Perform similarity search with enhanced contact information handling.
        
        Args:
            query: Search query string
            filter_dict: Optional metadata filters
            k: Number of results to return
            
        Returns:
            List[Document]: List of Document objects with content and metadata
        """
        if not self.collection:
            raise ValueError("Collection not initialized.")

        try:
            # Get relevant documents using compression retriever
            raw_results = self.compression_retriever.get_relevant_documents(query)
            
            # Process and filter results
            processed_results = []
            for result in raw_results:
                # Handle different result types
                if isinstance(result, Document):
                    doc = result
                elif isinstance(result, dict):
                    # Create Document from dictionary with enhanced metadata
                    doc = Document(
                        page_content=result.get('content', '') or result.get('page_content', ''),
                        metadata={
                            'type': result.get('metadata', {}).get('type', 'unknown'),
                            'class': result.get('metadata', {}).get('class', 'unknown'),
                            'email': result.get('metadata', {}).get('email', ''),
                            'parent_email': result.get('metadata', {}).get('parent_email', ''),
                            'head_teacher_email': result.get('metadata', {}).get('head_teacher_email', ''),
                            'subject': result.get('metadata', {}).get('subject', '')
                        }
                    )
                else:
                    # Create basic Document
                    doc = Document(
                        page_content=str(result),
                        metadata={}
                    )
                
                # Apply filters if provided
                if filter_dict:
                    if not all(doc.metadata.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                processed_results.append(doc)

            # Deduplicate while preserving Document structure
            seen_content = set()
            unique_results = []
            for doc in processed_results:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_results.append(doc)

            # Take top k results
            final_results = unique_results[:k]
            print(f"Found {len(final_results)} results:")
            
            if not final_results:
                print("No matching results found.")
                return []
            
            return final_results

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _print_store_summary(self, metadatas: List[Dict]):
        """Print store summary with contact information."""
        try:
            types = [m.get('type', 'unknown') for m in metadatas]
            emails = len([m for m in metadatas if m.get('email') or m.get('parent_email') or m.get('head_teacher_email')])
            
            print(f"\nVector store contents:")
            print(f"- Total documents: {len(types)}")
            print(f"- Teachers: {types.count('teacher')}")
            print(f"- Classes: {types.count('class')}")
            print(f"- Students: {types.count('student')}")
            print(f"- Documents with contact info: {emails}")
        except Exception as e:
            print(f"Error generating summary: {str(e)}")

def main():
    """Test the enhanced vector store."""
    try:
        print("\nInitializing vector store...")
        vector_store = VectorStore()

        print("\nSetting up vector store from embeddings...")
        embeddings_file = "data/embeddings.json"
        vector_store.create_vector_store(embeddings_file)

        # Test queries with contact information
        test_queries = [
            "What is Sarah Chen's email address?",
            "How can I contact the parents of students in Class 3A?",
            "What are the art teacher's office hours?",
            "Find contact information for the head teacher of Class 3B"
        ]

        print("\nTesting search functionality...")
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                results = vector_store.similarity_search(query)
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. Content: {doc.page_content}")
                    print(f"   Metadata: {doc.metadata}")
            except Exception as e:
                print(f"Error processing query: {str(e)}")

    except Exception as e:
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()