from openai import OpenAI
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI base URL
API_BASE_URL = "http://localhost:8000/academic-api"

# Email configuration
SMTP_SERVER = "smtp.gmail.com"  # Replace with your SMTP server
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")  # Your email address
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")  # Your email password

# Function to call the FastAPI endpoint
def call_api(endpoint: str, params: dict):
    """Call the FastAPI endpoint and return the response."""
    url = f"{API_BASE_URL}/{endpoint}/{params['class_name']}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed: {response.status_code}")

# Function to send emails
def send_email(to_email: str, subject: str, body: str):
    """Send an email using SMTP."""
    try:
        # Create the email
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, message.as_string())
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function schemas for OpenAI
functions = [
    {
        "name": "get_students_by_class",
        "description": "Get a list of students in a specific class, including their email addresses and parent contact information.",
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "The class identifier (e.g., 3A, 3B, 3C)."
                }
            },
            "required": ["class_name"]
        }
    },
    {
        "name": "get_teacher_info",
        "description": "Get information about the teacher responsible for a specific class, including their email and office hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "The class identifier (e.g., 3A, 3B, 3C)."
                }
            },
            "required": ["class_name"]
        }
    }
]

# Main function to interact with OpenAI
def chat_with_gpt(query: str):
    """Process a query using OpenAI's function calling."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}],
        functions=functions,
        function_call="auto"
    )

    # Check if the model wants to call a function
    if response.choices[0].message.function_call:
        function_call = response.choices[0].message.function_call
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)

        # Call the function
        if function_name == "get_students_by_class":
            students = call_api("students", arguments)
            return students
        elif function_name == "get_teacher_info":
            teacher = call_api("teacher", arguments)
            return teacher
    else:
        return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # Query OpenAI to get students in class 3A
    query = "Get the list of students in class 3A and their parent email addresses."
    students = chat_with_gpt(query)

    # Send emails to parents
    for student in students:
        parent_email = student["parent_email"]
        subject = "Upcoming Parent-Teacher Meeting"
        body = f"Dear Parent, this is a reminder about the upcoming parent-teacher meeting for {student['name']}."
        send_email(parent_email, subject, body)

    # Query OpenAI to get the teacher for class 3A
    query = "Get the teacher for class 3A and their email address."
    teacher = chat_with_gpt(query)

    # Send email to the teacher
    teacher_email = teacher["email"]
    subject = "Reminder: Upcoming Meeting"
    body = f"Dear {teacher['name']}, this is a reminder about the upcoming meeting."
    send_email(teacher_email, subject, body)