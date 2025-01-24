import json
import os
from typing import Dict, List, Any
from datetime import datetime

class DataProcessor:
    def __init__(self):
        """Initialize with academic data structure"""
        self.academic_data = {
            "teachers": [
                {
                    "name": "Sarah Chen",
                    "age": 32,
                    "subject": "Mathematics",
                    "class_responsibility": "3A",
                    "characteristics": [
                        "Known for using innovative technology in teaching",
                        "Organizes annual mathematics competitions"
                    ]
                },
                {
                    "name": "James Liu",
                    "age": 45,
                    "subject": "Science",
                    "class_responsibility": "3B",
                    "characteristics": [
                        "Published several science textbooks",
                        "Leads the school's science club"
                    ]
                },
                {
                    "name": "Emily Wang",
                    "age": 38,
                    "subject": "Art",
                    "class_responsibility": "3C",
                    "characteristics": [
                        "Professional painter",
                        "Organizes annual art exhibitions"
                    ]
                }
            ],
            "classes": [
                {
                    "name": "3A",
                    "location": "Third floor, East Wing",
                    "special_feature": "Excellence in Mathematics and Technology",
                    "head_teacher": "Ms. Chen",
                    "additional_info": "Known for winning regional math competitions"
                },
                {
                    "name": "3B",
                    "location": "Third floor, Central Wing",
                    "special_feature": "Focus on Science and Research",
                    "head_teacher": "Mr. Liu",
                    "additional_info": "Has a dedicated science laboratory"
                },
                {
                    "name": "3C",
                    "location": "Third floor, West Wing",
                    "special_feature": "Strong Arts Program",
                    "head_teacher": "Mrs. Wang",
                    "additional_info": "Regular art exhibitions and performances"
                }
            ],
            "students": [
                {
                    "name": "David Zhang",
                    "age": 14,
                    "class": "3A",
                    "address": "123 Maple Street, East District"
                },
                {
                    "name": "Linda Wu",
                    "age": 15,
                    "class": "3A",
                    "address": "456 Oak Road, North District"
                },
                {
                    "name": "Michael Chen",
                    "age": 14,
                    "class": "3A",
                    "address": "789 Pine Avenue, Central District"
                },
                {
                    "name": "Sophie Liu",
                    "age": 15,
                    "class": "3B",
                    "address": "321 Cedar Lane, West District"
                },
                {
                    "name": "Thomas Yang",
                    "age": 14,
                    "class": "3B",
                    "address": "654 Birch Street, South District"
                },
                {
                    "name": "Emma Sun",
                    "age": 15,
                    "class": "3B",
                    "address": "987 Elm Road, East District"
                },
                {
                    "name": "Kevin Wang",
                    "age": 14,
                    "class": "3B",
                    "address": "147 Spruce Avenue, North District"
                },
                {
                    "name": "Rachel Li",
                    "age": 15,
                    "class": "3C",
                    "address": "258 Willow Lane, Central District"
                },
                {
                    "name": "Jason Huang",
                    "age": 14,
                    "class": "3C",
                    "address": "369 Palm Street, West District"
                },
                {
                    "name": "Amy Zhou",
                    "age": 15,
                    "class": "3C",
                    "address": "741 Cherry Road, South District"
                }
            ]
        }

    def save_to_json(self, output_dir: str = "data") -> str:
        """Save academic data to JSON file"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output path
        output_path = os.path.join(output_dir, "academic_data.json")
        
        # Add metadata
        data_with_metadata = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "data_structure": "Academic Management System"
            },
            "data": self.academic_data
        }
        
        # Save with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_with_metadata, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def get_class_stats(self) -> Dict[str, Any]:
        """Get statistics for each class"""
        stats = {}
        for class_info in self.academic_data["classes"]:
            class_name = class_info["name"]
            students = [s for s in self.academic_data["students"] 
                       if s["class"] == class_name]
            
            stats[class_name] = {
                "total_students": len(students),
                "average_age": sum(s["age"] for s in students) / len(students),
                "teacher": next(t["name"] for t in self.academic_data["teachers"] 
                              if t["class_responsibility"] == class_name)
            }
        
        return stats
    
    def get_students_by_class(self, class_name: str) -> List[Dict]:
        """Get all students in a specific class"""
        return [s for s in self.academic_data["students"] 
                if s["class"] == class_name]
    
    def get_teacher_info(self, class_name: str) -> Dict:
        """Get teacher information for a specific class"""
        return next((t for t in self.academic_data["teachers"] 
                    if t["class_responsibility"] == class_name), None)

def main():
    """Main function to demonstrate data processor usage"""
    processor = DataProcessor()
    
    # Save data to JSON
    output_path = processor.save_to_json()
    print(f"Data saved to: {output_path}")
    
    # Print class statistics
    print("\nClass Statistics:")
    stats = processor.get_class_stats()
    for class_name, class_stats in stats.items():
        print(f"\nClass {class_name}:")
        print(f"Teacher: {class_stats['teacher']}")
        print(f"Total Students: {class_stats['total_students']}")
        print(f"Average Age: {class_stats['average_age']:.1f}")
        
        # Print students in this class
        students = processor.get_students_by_class(class_name)
        print("\nStudents:")
        for student in students:
            print(f"- {student['name']} (Age: {student['age']})")

if __name__ == "__main__":
    main()
