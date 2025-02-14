from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Academic API", description="API for managing academic data", version="1.0")

# Create a router with a base path
router = APIRouter(prefix="/academic-api")

# Sample academic data
academic_data = {
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
            "name": "Sophie Liu",
            "age": 15,
            "email": "sophie.liu@school.edu",
            "class": "3B",
            "address": "321 Cedar Lane, West District",
            "parent_email": "parent.liu@email.com"
        }
    ]
}

# Helper functions to retrieve data
def get_students_by_class(class_name: str) -> List[Dict]:
    """Get all students in a specific class."""
    return [s for s in academic_data["students"] if s["class"] == class_name]

def get_teacher_info(class_name: str) -> Dict:
    """Get the teacher responsible for a specific class."""
    return next((t for t in academic_data["teachers"] if t["class_responsibility"] == class_name), None)

def get_class_info(class_name: str) -> Dict:
    """Get information about a specific class."""
    return next((c for c in academic_data["classes"] if c["name"] == class_name), None)

# Root endpoint
@router.get("/")
def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the Academic API!"}

# Students endpoint
@router.get("/students/{class_name}", response_model=List[Dict])
def get_students(class_name: str):
    """Get all students in a specific class."""
    students = get_students_by_class(class_name)
    if not students:
        raise HTTPException(status_code=404, detail=f"No students found in class {class_name}")
    return students

# Teacher endpoint
@router.get("/teacher/{class_name}", response_model=Dict)
def get_teacher(class_name: str):
    """Get the teacher responsible for a specific class."""
    teacher = get_teacher_info(class_name)
    if not teacher:
        raise HTTPException(status_code=404, detail=f"No teacher found for class {class_name}")
    return teacher

# Class endpoint
@router.get("/class/{class_name}", response_model=Dict)
def get_class(class_name: str):
    """Get information about a specific class."""
    class_info = get_class_info(class_name)
    if not class_info:
        raise HTTPException(status_code=404, detail=f"No information found for class {class_name}")
    return class_info

# Include the router in the app
app.include_router(router)

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)