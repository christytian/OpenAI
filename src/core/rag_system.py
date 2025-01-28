# src/core/rag_system.py
import dotenv
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .vector_store import VectorStore
import os

dotenv.load_dotenv()

class RAGSystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def setup(self, embeddings_file: str):
        # Create vector store and retriever
        chroma_store = self.vector_store.create_vector_store(embeddings_file)
        self.retriever = chroma_store.as_retriever(
            search_kwargs={"k": 30}
        )

        # Create system and human message templates
        system_template = """You are an academic database system. Use the following context to provide accurate information.

        Context: {context}

        Instructions:
        1. For Teacher Queries:
        - List each teacher's name, subject, class responsibility, and characteristics
        2. For Class Queries:
        - Show location, features, head teacher, and student count for each class
        3. For Student Queries:
        - List students with their full details (name, age, address)
        - Show total student count vs. found students

        Remember to only use information explicitly stated in the context."""

        human_template = "{question}"

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def _detect_query_type(self, question: str):
        """Detect the type of query."""
        question = question.lower()
        if any(word in question for word in ['teacher', 'teaches', 'instructor']):
            return 'teacher'
        elif any(word in question for word in ['class', 'classes', 'classroom']):
            return 'class'
        elif any(word in question for word in ['student', 'students', 'pupil']):
            return 'student'
        return None
    
    def _extract_class_from_query(self, question: str):
        """Extract class reference from the query."""
        # Look for Class 3A, 3B, 3C patterns
        match = re.search(r'Class\s*(3[A-C])', question, re.IGNORECASE)
        return match.group(1).upper() if match else None


    def query(self, question: str) -> str:
        """Perform a comprehensive query across the vector store."""
        if not hasattr(self, 'retriever'):
            raise ValueError("RAG system not set up. Call setup() first.")
        
        try:
            # Detect query type and potential class filter
            query_type = self._detect_query_type(question)
            class_filter = self._extract_class_from_query(question)
            
            # Prepare filters
            filters = {}
            if query_type:
                filters['type'] = query_type
            if class_filter:
                filters['class'] = class_filter
            
            # Perform similarity search with dynamic filtering
            print(f"\nQuerying with filters: {filters}")
            
            # Attempt type and class-specific search first
            type_docs = []
            if filters:
                try:
                    type_docs = self.vector_store.similarity_search(
                        question, 
                        filter_dict=filters, 
                        k=30
                    )
                except Exception as filter_error:
                    print(f"Filtered search error: {filter_error}")
            
            # Fallback to unfiltered search if no results
            if not type_docs:
                print("No results with filters. Performing unfiltered search.")
                type_docs = self.vector_store.similarity_search(question, k=30)
            
            # Deduplicate documents
            seen = set()
            unique_docs = []
            for doc in type_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            
            # Handle no results scenario
            if not unique_docs:
                return "No relevant information found."
            
            # Create context from unique documents
            context = "\n".join(doc.page_content for doc in unique_docs)
            
            # Format messages and get response
            messages = self.prompt.format_messages(
                context=context,
                question=question
            )
            
            # Generate response
            response = self.llm.invoke(messages)
            return response.content
        
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred: {str(e)}"


def main():
    # Example usage
    rag_system = RAGSystem()
    rag_system.setup("data/embeddings.json")
    
    # Test queries
    test_queries = [
        "Who teaches mathematics?",
        "List all students in Class 3C",
        "Tell me about Class 3A",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag_system.query(query)
        print(result)


if __name__ == "__main__":
    main()