import json
import os
from typing import Dict, List, Any
from datetime import datetime

class DataProcessor:
    def __init__(self):
        """Initialize with academic data structure including email information"""
        self.academic_data = {
            "teachers": [
                {
                    "name": "Sarah Chen",
                    "age": 32,
                    "email": "sarah.chen@school.edu",
                    "subject": "Mathematics",
                    "class_responsibility": "3A",
                    "office_hours": "Monday and Wednesday 3-4pm",
                    "characteristics": [
                        "Known for using innovative technology in teaching",
                        "Organizes annual mathematics competitions"
                    ]
                },
                {
                    "name": "James Liu",
                    "age": 45,
                    "email": "james.liu@school.edu",
                    "subject": "Science",
                    "class_responsibility": "3B",
                    "office_hours": "Tuesday and Thursday 3-4pm",
                    "characteristics": [
                        "Published several science textbooks",
                        "Leads the school's science club"
                    ]
                },
                {
                    "name": "Emily Wang",
                    "age": 38,
                    "email": "emily.wang@school.edu",
                    "subject": "Art",
                    "class_responsibility": "3C",
                    "office_hours": "Friday 2-4pm",
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
                    "head_teacher_email": "sarah.chen@school.edu",
                    "additional_info": "Known for winning regional math competitions"
                },
                {
                    "name": "3B",
                    "location": "Third floor, Central Wing",
                    "special_feature": "Focus on Science and Research",
                    "head_teacher": "Mr. Liu",
                    "head_teacher_email": "james.liu@school.edu",
                    "additional_info": "Has a dedicated science laboratory"
                },
                {
                    "name": "3C",
                    "location": "Third floor, West Wing",
                    "special_feature": "Strong Arts Program",
                    "head_teacher": "Mrs. Wang",
                    "head_teacher_email": "emily.wang@school.edu",
                    "additional_info": "Regular art exhibitions and performances"
                }
            ],
            "students": [
                {
                    "name": "David Zhang",
                    "age": 14,
                    "email": "david.zhang@school.edu",
                    "class": "3A",
                    "address": "123 Maple Street, East District",
                    "parent_email": "parent.zhang@email.com"
                },
                {
                    "name": "Linda Wu",
                    "age": 15,
                    "email": "linda.wu@school.edu",
                    "class": "3A",
                    "address": "456 Oak Road, North District",
                    "parent_email": "parent.wu@email.com"
                },
                {
                    "name": "Michael Chen",
                    "age": 14,
                    "email": "michael.chen@school.edu",
                    "class": "3A",
                    "address": "789 Pine Avenue, Central District",
                    "parent_email": "parent.chen@email.com"
                },
                {
                    "name": "Sophie Liu",
                    "age": 15,
                    "email": "sophie.liu@school.edu",
                    "class": "3B",
                    "address": "321 Cedar Lane, West District",
                    "parent_email": "parent.liu@email.com"
                },
                {
                    "name": "Thomas Yang",
                    "age": 14,
                    "email": "thomas.yang@school.edu",
                    "class": "3B",
                    "address": "654 Birch Street, South District",
                    "parent_email": "parent.yang@email.com"
                },
                {
                    "name": "Emma Sun",
                    "age": 15,
                    "email": "emma.sun@school.edu",
                    "class": "3B",
                    "address": "987 Elm Road, East District",
                    "parent_email": "parent.sun@email.com"
                },
                {
                    "name": "Kevin Wang",
                    "age": 14,
                    "email": "kevin.wang@school.edu",
                    "class": "3B",
                    "address": "147 Spruce Avenue, North District",
                    "parent_email": "parent.wang@email.com"
                },
                {
                    "name": "Rachel Li",
                    "age": 15,
                    "email": "rachel.li@school.edu",
                    "class": "3C",
                    "address": "258 Willow Lane, Central District",
                    "parent_email": "parent.li@email.com"
                },
                {
                    "name": "Jason Huang",
                    "age": 14,
                    "email": "jason.huang@school.edu",
                    "class": "3C",
                    "address": "369 Palm Street, West District",
                    "parent_email": "parent.huang@email.com"
                },
                {
                    "name": "Amy Zhou",
                    "age": 15,
                    "email": "amy.zhou@school.edu",
                    "class": "3C",
                    "address": "741 Cherry Road, South District",
                    "parent_email": "parent.zhou@email.com"
                }
            ]
        }

    def save_to_json(self, output_dir: str = "data") -> str:
        """Save academic data to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "academic_data.json")
        
        data_with_metadata = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "data_structure": "Academic Management System",
                "includes_email": True
            },
            "data": self.academic_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_with_metadata, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def get_class_stats(self) -> Dict[str, Any]:
        """Get statistics for each class including contact information"""
        stats = {}
        for class_info in self.academic_data["classes"]:
            class_name = class_info["name"]
            students = [s for s in self.academic_data["students"] 
                       if s["class"] == class_name]
            teacher = next(t for t in self.academic_data["teachers"] 
                         if t["class_responsibility"] == class_name)
            
            stats[class_name] = {
                "total_students": len(students),
                "average_age": sum(s["age"] for s in students) / len(students),
                "teacher": {
                    "name": teacher["name"],
                    "email": teacher["email"],
                    "office_hours": teacher["office_hours"]
                },
                "student_emails": [s["email"] for s in students],
                "parent_emails": [s["parent_email"] for s in students]
            }
        
        return stats
    
    def get_students_by_class(self, class_name: str) -> List[Dict]:
        """Get all students in a specific class with contact information"""
        return [s for s in self.academic_data["students"] 
                if s["class"] == class_name]
    
    def get_teacher_info(self, class_name: str) -> Dict:
        """Get teacher information including contact details for a specific class"""
        return next((t for t in self.academic_data["teachers"] 
                    if t["class_responsibility"] == class_name), None)
    
    def get_contact_list(self, class_name: str) -> Dict[str, List[str]]:
        """Get all contact information for a specific class"""
        teacher = self.get_teacher_info(class_name)
        students = self.get_students_by_class(class_name)
        
        return {
            "teacher_email": teacher["email"] if teacher else None,
            "student_emails": [s["email"] for s in students],
            "parent_emails": [s["parent_email"] for s in students],
            "all_emails": (
                [teacher["email"]] +
                [s["email"] for s in students] +
                [s["parent_email"] for s in students]
            ) if teacher else []
        }

def main():
    """Main function to demonstrate data processor usage"""
    processor = DataProcessor()
    
    # Save data to JSON
    output_path = processor.save_to_json()
    print(f"Data saved to: {output_path}")
    
    # Print class statistics with contact information
    print("\nClass Statistics and Contact Information:")
    stats = processor.get_class_stats()
    for class_name, class_stats in stats.items():
        print(f"\nClass {class_name}:")
        print(f"Teacher: {class_stats['teacher']['name']}")
        print(f"Teacher Email: {class_stats['teacher']['email']}")
        print(f"Office Hours: {class_stats['teacher']['office_hours']}")
        print(f"Total Students: {class_stats['total_students']}")
        print(f"Average Age: {class_stats['average_age']:.1f}")
        
        # Print students in this class
        students = processor.get_students_by_class(class_name)
        print("\nStudents:")
        for student in students:
            print(f"- {student['name']} (Age: {student['age']})")
            print(f"  Email: {student['email']}")
            print(f"  Parent Email: {student['parent_email']}")
        
        # Print contact list
        contacts = processor.get_contact_list(class_name)
        print(f"\nTotal email addresses: {len(contacts['all_emails'])}")

if __name__ == "__main__":
    main()