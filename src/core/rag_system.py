# src/core/rag_system.py
import dotenv
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

    def query(self, question: str) -> str:
        if not hasattr(self, 'retriever'):
            raise ValueError("RAG system not set up. Call setup() first.")
    
        try:
            print("\nQuerying different content types...")
            
            # Get teacher information
            teacher_docs = self.vector_store.similarity_search(
                question,
                k=5,
                filter_dict={"content_type": "teacher"}
            )
            
            # Get class information
            class_docs = self.vector_store.similarity_search(
                question,
                k=5,
                filter_dict={"content_type": "class"}
            )
            
            # Get student information
            student_docs = self.vector_store.similarity_search(
                question,
                k=10,
                filter_dict={"content_type": "student"}
            )
            
            # Remove duplicates while preserving order
            def deduplicate_docs(docs):
                seen = set()
                unique_docs = []
                for doc in docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        unique_docs.append(doc)
                return unique_docs
            
            # Deduplicate each category
            teacher_docs = deduplicate_docs(teacher_docs)
            class_docs = deduplicate_docs(class_docs)
            student_docs = deduplicate_docs(student_docs)
            
            # Build structured context
            context_parts = []
            
            if teacher_docs:
                context_parts.append("TEACHER INFORMATION:")
                context_parts.extend(doc.page_content for doc in teacher_docs)
            
            if class_docs:
                context_parts.append("\nCLASS INFORMATION:")
                context_parts.extend(doc.page_content for doc in class_docs)
            
            if student_docs:
                context_parts.append("\nSTUDENT INFORMATION:")
                context_parts.extend(doc.page_content for doc in student_docs)
            
            context = "\n".join(context_parts)
            
            # Debug print
            print("\nRetrieved Information:")
            print(f"Teachers: {len(teacher_docs)} unique documents")
            print(f"Classes: {len(class_docs)} unique documents")
            print(f"Students: {len(student_docs)} unique documents")
            
            # Get response
            messages = self.prompt.format_messages(
                context=context,
                question=question
            )
            response = self.llm.invoke(messages)
            return response.content
                
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred: {str(e)}"

def main():
    try:
        rag = RAGSystem()
        rag.setup("data/embeddings.json")
        
        questions = [
            "List each teacher with their subject and characteristics.",
            "For each class (3A, 3B, 3C), provide complete information about location, teacher, and students.",
            "Tell me about all students in Class 3B.",
            "Give me a complete breakdown of Class 3A including teacher, location, features, and all students."
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            answer = rag.query(question)
            print(f"A: {answer}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()