from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Union
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware
import json

# Load .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.7
)

# Logs
user_logs = defaultdict(lambda: {"questions": 0, "types": defaultdict(int)})

# Schemas
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

# Type classification
def classify_question(msg: str) -> str:
    msg = msg.lower()
    if any(w in msg for w in ["tree", "graph", "array", "leetcode", "code", "function", "bug", "solution", "runtime", "compile"]):
        return "Coding"
    elif "interview" in msg or "tell me about yourself" in msg:
        return "HR"
    else:
        return "General"

# Checks
def looks_like_answer(msg: str) -> bool:
    return len(msg.split()) > 10 and any(x in msg.lower() for x in ["i ", "my ", "we ", "our ", "project", "experience", "challenge"])

def looks_like_code(msg: str) -> bool:
    return "def " in msg or "class " in msg or "public static" in msg or "#include" in msg or "{" in msg or "```" in msg

@app.post("/agent-chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest):
    message = request.message.strip()
    user_id = request.user_id
    qtype = classify_question(message)
    user_logs[user_id]["questions"] += 1
    user_logs[user_id]["types"][qtype] += 1

    # Standard response prompt
    default_prompt = PromptTemplate.from_template(
        """
        You are an AI Interview Coach. Help users with coding, HR, and DSA questions.

        - Answer coding or DSA questions clearly.
        - If it's a user's answer, give feedback.
        - If it's just a chat message, respond helpfully.
        - Format responses in Markdown.

        Message: {prompt}
        """
    )

    response = (default_prompt | llm).invoke({"prompt": message})
    reply = response.content

    # Init extra metrics
    result = {
        "structure_score": None, "clarity_score": None, "tone_score": None,
        "relevance_score": None, "logic_score": None, "code_clarity": None,
        "efficiency_score": None, "best_practices": None,
        "detected_issues": [], "suggestions": []
    }

    # Answer evaluation
    if looks_like_answer(message):
        eval_prompt = PromptTemplate.from_template(
            """
            You are an expert Interview Answer Evaluator. Analyze the user's answer and return this JSON:

            {{
                "structure": 0-100,
                "clarity": 0-100,
                "tone": 0-100,
                "relevance": 0-100,
                "issues": ["..."],
                "suggestions": ["..."]
            }}

            User Answer: {answer}
            """
        )
        try:
            eval_response = (eval_prompt | llm).invoke({"answer": message})
            data = json.loads(eval_response.content)
            result.update({
                "structure_score": data.get("structure"),
                "clarity_score": data.get("clarity"),
                "tone_score": data.get("tone"),
                "relevance_score": data.get("relevance"),
                "detected_issues": data.get("issues", []),
                "suggestions": data.get("suggestions", [])
            })
        except Exception as e:
            result["suggestions"].append("Couldn't evaluate answer. Try rewriting it clearly.")

    # Code evaluation
    elif looks_like_code(message):
        code_prompt = PromptTemplate.from_template(
            """
            You are a coding assistant. Analyze the user's code and return this JSON:

            {{
                "logic_score": 0-100,
                "code_clarity": 0-100,
                "efficiency_score": 0-100,
                "best_practices": 0-100,
                "issues": ["..."],
                "suggestions": ["..."]
            }}

            Code:\n{code}
            """
        )
        try:
            eval_response = (code_prompt | llm).invoke({"code": message})
            data = json.loads(eval_response.content)
            result.update({
                "logic_score": data.get("logic_score"),
                "code_clarity": data.get("code_clarity"),
                "efficiency_score": data.get("efficiency_score"),
                "best_practices": data.get("best_practices"),
                "detected_issues": data.get("issues", []),
                "suggestions": data.get("suggestions", [])
            })
        except Exception as e:
            result["suggestions"].append("Code parsing failed. Try formatting it better.")

    return ChatResponse(
        reply=reply,
        total_questions=user_logs[user_id]["questions"],
        question_types=dict(user_logs[user_id]["types"]),
        **result
    )
