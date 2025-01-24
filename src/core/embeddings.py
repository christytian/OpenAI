from langchain.embeddings import OpenAIEmbeddings
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
        """Load and format academic data from JSON"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract relevant data and format into text chunks
        texts = []
        
        # Process teachers
        for teacher in data['data']['teachers']:
            text = (
                f"Teacher Information: {teacher['name']} is {teacher['age']} years old "
                f"and teaches {teacher['subject']}. They are responsible for class "
                f"{teacher['class_responsibility']}. Notable characteristics: "
                f"{'; '.join(teacher['characteristics'])}."
            )
            texts.append(text)
        
        # Process classes
        for class_info in data['data']['classes']:
            text = (
                f"Class Information: {class_info['name']} is located in the "
                f"{class_info['location']}. The class has {class_info['special_feature']} "
                f"and {class_info['additional_info']}. The head teacher is "
                f"{class_info['head_teacher']}."
            )
            texts.append(text)
        
        # Process students
        for student in data['data']['students']:
            text = (
                f"Student Information: {student['name']} is {student['age']} years old, "
                f"belongs to Class {student['class']}, and lives at {student['address']}."
            )
            texts.append(text)
        
        return texts
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for the provided texts"""
        embeddings = []
        for text in texts:
            embedding = self.embeddings.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def process_and_save_embeddings(self, input_file: str, output_file: str = "data/embeddings.json"):
        """Process academic data and save embeddings"""
        # Load and format texts
        texts = self.load_academic_data(input_file)
        
        # Create embeddings
        embeddings_data = []
        for i, text in enumerate(texts):
            embedding = self.embeddings.embed_query(text)
            embeddings_data.append({
                "id": f"text_{i}",
                "text": text,
                "embedding": embedding
            })
        
        # Save embeddings
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "model": "text-embedding-3-small",
                    "version": "1.0"
                },
                "embeddings": embeddings_data
            }, f, indent=2)
        
        return output_file

def main():
    """Main function to demonstrate embeddings creation"""
    manager = EmbeddingsManager()
    
    # Process and save embeddings
    input_file = "data/academic_data.json"
    output_file = "data/embeddings.json"
    
    try:
        saved_file = manager.process_and_save_embeddings(input_file, output_file)
        print(f"Embeddings saved to: {saved_file}")
    except Exception as e:
        print(f"Error processing embeddings: {e}")

if __name__ == "__main__":
    main()