from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Add CORS middleware to allow frontend calls (e.g., from bolt.new)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory user log
user_logs = defaultdict(lambda: {"questions": 0, "types": defaultdict(int)})

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.7
)

# Request & response schemas
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    total_questions: int
    question_types: dict

# Basic classifier
def classify_question(msg: str) -> str:
    msg = msg.lower()
    if "tree" in msg or "graph" in msg or "array" in msg:
        return "DSA"
    elif "interview" in msg or "tell me about yourself" in msg:
        return "HR"
    elif "code" in msg or "program" in msg:
        return "Coding"
    else:
        return "General"

@app.post("/agent-chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest):
    message = request.message.strip()
    user_id = request.user_id

    # If message is empty, generate a random interview question
    if not message:
        message = "Generate a random interview question from DSA, HR, or general topics."

    # Update logs
    qtype = classify_question(message)
    user_logs[user_id]["questions"] += 1
    user_logs[user_id]["types"][qtype] += 1

    # Prompt template
    prompt = PromptTemplate.from_template("""
        You are an AI Interview Coach. Understand this user message and respond clearly.

        - If it's a question, answer it like an expert.
        - If it's an answer attempt by the user, give feedback.
        - Keep it brief, helpful, and formatted in Markdown.

        User: {prompt}
    """)

    # Generate response
    response = (prompt | llm).invoke({"prompt": message})

    return ChatResponse(
        reply=response.content,
        total_questions=user_logs[user_id]["questions"],
        question_types=dict(user_logs[user_id]["types"])
    )
