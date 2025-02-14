from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class EmbeddingsManager:
    def __init__(self):
        """Initialize embeddings manager with OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_academic_data(self, file_path: str) -> List[str]:
        """Load and format academic data from JSON with email information"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        texts_with_metadata = []
        
        # Process teachers with email and office hours
        for teacher in data['data']['teachers']:
            text = (
                f"Teacher Information: {teacher['name']} is {teacher['age']} years old "
                f"and teaches {teacher['subject']}. They are responsible for class "
                f"{teacher['class_responsibility']}. Their email is {teacher['email']} "
                f"and office hours are {teacher['office_hours']}. Notable characteristics: "
                f"{'; '.join(teacher['characteristics'])}."
            )
            texts_with_metadata.append({
                "text": text,
                "metadata": {
                    "type": "teacher",
                    "class": teacher['class_responsibility'],
                    "email": teacher['email'],
                    "subject": teacher['subject']
                }
            })
        
        # Process classes with head teacher email
        for class_info in data['data']['classes']:
            student_count = len([s for s in data['data']['students'] 
                            if s['class'] == class_info['name']])
            text = (
                f"Class Information: {class_info['name']} is located in the "
                f"{class_info['location']}. The class has {class_info['special_feature']} "
                f"and {class_info['additional_info']}. The head teacher is "
                f"{class_info['head_teacher']} (email: {class_info['head_teacher_email']}). "
                f"This class has {student_count} students."
            )
            texts_with_metadata.append({
                "text": text,
                "metadata": {
                    "type": "class",
                    "class": class_info['name'],
                    "head_teacher_email": class_info['head_teacher_email']
                }
            })
        
        # Process students with email and parent contact
        for student in data['data']['students']:
            text = (
                f"Student Information: {student['name']} is {student['age']} years old, "
                f"belongs to Class {student['class']}, and lives at {student['address']}. "
                f"Student email: {student['email']}. Parent contact: {student['parent_email']}."
            )
            texts_with_metadata.append({
                "text": text,
                "metadata": {
                    "type": "student",
                    "class": student['class'],
                    "email": student['email'],
                    "parent_email": student['parent_email']
                }
            })
        
        # Debug: Print generated texts
        print(f"\nTotal texts with metadata generated: {len(texts_with_metadata)}")
        for idx, item in enumerate(texts_with_metadata[:5]):
            print(f"\nText {idx + 1}: {item['text']}")
            print(f"Metadata {idx + 1}: {item['metadata']}")
        
        return texts_with_metadata
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for the provided texts with enhanced metadata"""
        embeddings_data = []
        for i, item in enumerate(texts):
            embedding = self.embeddings.embed_query(item["text"])
            embeddings_data.append({
                "id": f"text_{i}",
                "text": item["text"],
                "embedding": embedding,
                "metadata": {
                    "type": item["metadata"].get("type", "unknown"),
                    "class": item["metadata"].get("class", "unknown"),
                    "email": item["metadata"].get("email", ""),
                    "parent_email": item["metadata"].get("parent_email", ""),
                    "head_teacher_email": item["metadata"].get("head_teacher_email", ""),
                    "subject": item["metadata"].get("subject", "")
                }
            })
        return embeddings_data
    
    def process_and_save_embeddings(self, input_file: str, output_file: str = "data/embeddings.json"):
        """Process academic data and save embeddings with enhanced metadata"""
        # Load and format texts with metadata
        texts_with_metadata = self.load_academic_data(input_file)
        
        # Create embeddings
        embeddings_data = self.create_embeddings(texts_with_metadata)
        
        # Save embeddings
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "model": "text-embedding-3-small",
                    "version": "1.0",
                    "includes_contact_info": True,
                    "contact_types": ["email", "parent_email", "head_teacher_email"]
                },
                "embeddings": embeddings_data
            }, f, indent=2)
        
        print(f"Embeddings saved with enhanced metadata to: {output_file}")
        return output_file

def main():
    """Test the enhanced embeddings manager"""
    try:
        manager = EmbeddingsManager()
        
        # Process and save embeddings
        input_file = "data/academic_data.json"
        output_file = "data/embeddings.json"
        
        saved_file = manager.process_and_save_embeddings(input_file, output_file)
        print(f"Enhanced embeddings saved to: {saved_file}")
        
        # Verify the saved embeddings
        with open(saved_file, 'r') as f:
            saved_data = json.load(f)
            print("\nVerification:")
            print(f"Total embeddings: {len(saved_data['embeddings'])}")
            print("Metadata includes:", saved_data['metadata'])
            
            # Show sample embedding metadata
            if saved_data['embeddings']:
                print("\nSample embedding metadata:")
                print(json.dumps(saved_data['embeddings'][0]['metadata'], indent=2))
        
    except Exception as e:
        print(f"Error processing embeddings: {e}")

if __name__ == "__main__":
    main()