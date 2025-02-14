from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from .enhanced_vector_store import VectorStore
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        """Initialize RAG system with enhanced vector store and language model."""
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Enhanced system template with contact information handling
        system_template = """You are an advanced academic database assistant. Use the following retrieved information to answer questions.
        Only use information explicitly mentioned in the retrieved documents. If information is not available, say so.
        Present the information in a clear, organized manner.

        Retrieved Information:
        {context}

        When responding:
        1. Be comprehensive and include all relevant details
        2. Present information in a well-structured format
        3. If multiple records exist (e.g., multiple students), list them all
        4. Include contact information when available:
           - Teacher email addresses and office hours
           - Student email addresses and parent contacts
           - Class head teacher contact information
        5. Group related information together (e.g., all contact methods for one person)
        6. For class information:
           - Location and features
           - Head teacher and their contact details
           - Student composition and contact details
        7. Format emails and contact information in an easily readable way
        8. When listing multiple contacts, use clear sections and bullet points
        """

        human_template = "Question: {question}"

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # Define response templates for different query types
        self.response_templates = {
            'teacher': """
Teacher Information:
- Name: {name}
- Subject: {subject}
- Class Responsibility: {class_responsibility}
- Contact Information:
  * Email: {email}
  * Office Hours: {office_hours}
- Characteristics:
{characteristics}
            """,
            'student': """
Student Information:
- Name: {name}
- Age: {age}
- Class: {class}
- Contact Information:
  * Student Email: {email}
  * Parent Contact: {parent_email}
- Address: {address}
            """,
            'class': """
Class Information:
- Class: {name}
- Location: {location}
- Special Features: {features}
- Head Teacher: {head_teacher}
  * Email: {head_teacher_email}
- Additional Information: {additional_info}
- Student Count: {student_count}
            """
        }

    def setup(self, embeddings_file: str):
        """Initialize the vector store with embeddings."""
        try:
            self.vector_store.create_vector_store(embeddings_file)
            print("Enhanced RAG system initialization complete.")
        except Exception as e:
            print(f"Error during setup: {e}")
            raise

    def _create_context(self, documents) -> str:
        """Create enhanced context from retrieved documents with contact information."""
        context_parts = []
        for doc in documents:
            # Get metadata
            doc_type = doc.metadata.get('type', 'unknown').upper()
            metadata_str = ""
            
            # Add relevant metadata based on document type
            if doc.metadata.get('email'):
                metadata_str += f"\nContact: {doc.metadata['email']}"
            if doc.metadata.get('parent_email'):
                metadata_str += f"\nParent Contact: {doc.metadata['parent_email']}"
            if doc.metadata.get('head_teacher_email'):
                metadata_str += f"\nHead Teacher Contact: {doc.metadata['head_teacher_email']}"
            if doc.metadata.get('subject'):
                metadata_str += f"\nSubject: {doc.metadata['subject']}"
            
            # Combine content and metadata
            context_parts.append(
                f"[{doc_type}] {doc.page_content}{metadata_str}"
            )
        
        return "\n\n".join(context_parts)

    def _format_response(self, question: str, context: str, response: str) -> str:
        """Format the response with proper structure and contact information."""
        # Add a summary of available contact information
        contact_info = []
        for line in context.split('\n'):
            if any(term in line.lower() for term in ['email', 'contact', 'office hours']):
                contact_info.append(line.strip())
        
        if contact_info:
            response += "\n\nContact Information Available:\n" + "\n".join(
                f"- {info}" for info in contact_info
            )
        
        return response

    def query(self, question: str) -> str:
        """Process a query and return enhanced response with contact information."""
        try:
            # Get relevant documents with contact information
            documents = self.vector_store.similarity_search(
                query=question,
                k=5  # Adjust based on need
            )
            
            if not documents:
                return "No relevant information found."
            
            # Create enhanced context
            context = self._create_context(documents)
            
            # Format messages
            messages = self.prompt.format_messages(
                context=context,
                question=question
            )
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Format response with contact information
            formatted_response = self._format_response(
                question=question,
                context=context,
                response=response.content
            )
            
            return formatted_response

        except Exception as e:
            print(f"Error processing query: {e}")
            return f"An error occurred while processing your query: {str(e)}"


def main():
    """Test the enhanced RAG system."""
    try:
        # Initialize and setup
        rag = RAGSystem()
        rag.setup("data/embeddings.json")
        
        # Test queries with contact information focus
        test_queries = [
            "What is Sarah Chen's email and office hours?",
            "How can I contact the parents of students in Class 3A?",
            "Tell me about the art teacher and their contact information",
            "What are the contact details for Class 3B's head teacher?",
            "List all students in Class 3A with their email addresses",
            "Who should I contact about the science laboratory schedule?"
        ]
        
        # Process queries
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            result = rag.query(query)
            print(result)
            print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()