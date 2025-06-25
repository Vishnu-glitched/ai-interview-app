from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Union
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from collections import defaultdict
import os
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# User log for analytics
user_logs = defaultdict(lambda: {"questions": 0, "types": defaultdict(int)})

# Request and Response Schemas
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    total_questions: int
    question_types: Dict[str, int]
    structure_score: Union[int, None] = None
    clarity_score: Union[int, None] = None
    tone_score: Union[int, None] = None
    relevance_score: Union[int, None] = None
    logic_score: Union[int, None] = None
    code_clarity: Union[int, None] = None
    efficiency_score: Union[int, None] = None
    best_practices: Union[int, None] = None
    detected_issues: List[str] = []
    suggestions: List[str] = []

# Helpers
def classify_question(msg: str) -> str:
    msg = msg.lower()
    if any(w in msg for w in ["tree", "graph", "array", "leetcode", "code", "bug", "function"]):
        return "Coding"
    elif "interview" in msg or "yourself" in msg or "team" in msg:
        return "HR"
    return "General"

def looks_like_answer(msg: str) -> bool:
    return len(msg.split()) > 10 and any(x in msg.lower() for x in ["i ", "my ", "we ", "project", "experience", "challenge"])

def looks_like_code(msg: str) -> bool:
    return any(x in msg for x in ["def ", "class ", "#include", "public static", "{", "```", "System.out"])

# Main Chat Endpoint
@app.post("/agent-chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest):
    message = request.message.strip()
    user_id = request.user_id
    qtype = classify_question(message)
    
    user_logs[user_id]["questions"] += 1
    user_logs[user_id]["types"][qtype] += 1

    result = {
        "structure_score": None, "clarity_score": None, "tone_score": None,
        "relevance_score": None, "logic_score": None, "code_clarity": None,
        "efficiency_score": None, "best_practices": None,
        "detected_issues": [], "suggestions": []
    }

    # Step 1: Main reply
    chat_prompt = f"""You are an AI Interview Coach. Help users with coding, HR, or general queries. If the message is a question, give a short helpful reply. If it's an answer or code, just acknowledge and await evaluation.

Message: {message}
"""
    reply = model.generate_content(chat_prompt).text.strip()

    try:
        # Step 2: Evaluate interview answer
        if looks_like_answer(message):
            eval_prompt = f"""
Evaluate this interview answer and return a JSON with structure, clarity, tone, relevance scores (0-100), detected issues and suggestions.

Answer:
{message}

Return format:
{{
  "structure": 0-100,
  "clarity": 0-100,
  "tone": 0-100,
  "relevance": 0-100,
  "issues": ["..."],
  "suggestions": ["..."]
}}
"""
            response = model.generate_content(eval_prompt)
            data = json.loads(response.text)
            result.update({
                "structure_score": data.get("structure"),
                "clarity_score": data.get("clarity"),
                "tone_score": data.get("tone"),
                "relevance_score": data.get("relevance"),
                "detected_issues": data.get("issues", []),
                "suggestions": data.get("suggestions", [])
            })

        # Step 3: Evaluate code
        elif looks_like_code(message):
            eval_prompt = f"""
Evaluate this code and return JSON scores for logic, code_clarity, efficiency, and best_practices (0-100), along with detected issues and suggestions.

Code:
{message}

Return format:
{{
  "logic_score": 0-100,
  "code_clarity": 0-100,
  "efficiency_score": 0-100,
  "best_practices": 0-100,
  "issues": ["..."],
  "suggestions": ["..."]
}}
"""
            response = model.generate_content(eval_prompt)
            data = json.loads(response.text)
            result.update({
                "logic_score": data.get("logic_score"),
                "code_clarity": data.get("code_clarity"),
                "efficiency_score": data.get("efficiency_score"),
                "best_practices": data.get("best_practices"),
                "detected_issues": data.get("issues", []),
                "suggestions": data.get("suggestions", [])
            })

    except Exception as e:
        result["suggestions"].append("Evaluation failed. Try simplifying your input.")

    return ChatResponse(
        reply=reply,
        total_questions=user_logs[user_id]["questions"],
        question_types=dict(user_logs[user_id]["types"]),
        **result
    )
