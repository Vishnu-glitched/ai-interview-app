# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Union
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from collections import defaultdict
# from dotenv import load_dotenv
# import os
# import json
# import re

# # Load API key
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Use specific frontend domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Chat model (Groq LLaMA 3)
# llm = ChatGroq(
#     groq_api_key=GROQ_API_KEY,
#     model_name="llama3-70b-8192",
#     temperature=0.7,
# )

# # In-memory user logs
# user_logs = defaultdict(lambda: {"questions": 0, "types": defaultdict(int)})

# # Request/Response Models
# class ChatRequest(BaseModel):
#     user_id: str
#     message: str

# class ChatResponse(BaseModel):
#     reply: str
#     total_questions: int
#     question_types: Dict[str, int]
#     structure_score: Union[int, None] = None
#     clarity_score: Union[int, None] = None
#     tone_score: Union[int, None] = None
#     relevance_score: Union[int, None] = None
#     logic_score: Union[int, None] = None
#     code_clarity: Union[int, None] = None
#     efficiency_score: Union[int, None] = None
#     best_practices: Union[int, None] = None
#     detected_issues: List[str] = []
#     suggestions: List[str] = []

# # Utility: Classify message type
# def classify_question(msg: str) -> str:
#     msg = msg.lower()
#     if any(x in msg for x in ["tree", "graph", "leetcode", "array", "function", "runtime", "code"]):
#         return "Coding"
#     elif "interview" in msg or "yourself" in msg:
#         return "HR"
#     else:
#         return "General"

# # Check if it's likely an answer
# def looks_like_answer(msg: str) -> bool:
#     return len(msg.split()) > 10 and any(w in msg.lower() for w in ["i ", "my ", "project", "experience", "we "])

# # Check if it's likely code
# def looks_like_code(msg: str) -> bool:
#     return any(w in msg.lower() for w in ["def ", "class ", "#include", "public static", "```", "{"])

# # Helper to extract JSON from messy output
# def extract_json(text: str) -> dict:
#     try:
#         json_str = re.search(r"{.*}", text, re.DOTALL).group()
#         return json.loads(json_str)
#     except:
#         return {}

# @app.post("/agent-chat", response_model=ChatResponse)
# def agent_chat(request: ChatRequest):
#     message = request.message.strip()
#     user_id = request.user_id

#     # Fallback to random question if blank
#     if not message:
#         message = "Generate a random interview question from DSA, HR, or general topics."

#     qtype = classify_question(message)
#     user_logs[user_id]["questions"] += 1
#     user_logs[user_id]["types"][qtype] += 1

#     # Default reply
#     prompt = PromptTemplate.from_template("""
#     You are an AI Interview Coach. Help users with HR, coding, or DSA interview prep.

#     - If they ask a question, answer like an expert.
#     - If it's their answer, evaluate and give tips.
#     - If it's code, review and suggest improvements.
#     - Keep it in clear Markdown.

#     Message: {prompt}
#     """)
#     reply_text = (prompt | llm).invoke({"prompt": message}).content

#     # Initialize metrics
#     result = {
#         "structure_score": None, "clarity_score": None, "tone_score": None,
#         "relevance_score": None, "logic_score": None, "code_clarity": None,
#         "efficiency_score": None, "best_practices": None,
#         "detected_issues": [], "suggestions": []
#     }

#     # Evaluate answer
#     try:
#         if looks_like_answer(message):
#             eval_prompt = PromptTemplate.from_template("""
#             You are an interview coach. Evaluate the candidate's answer.
#             Respond ONLY in this JSON:

#             {{
#                 "structure": 0-100,
#                 "clarity": 0-100,
#                 "tone": 0-100,
#                 "relevance": 0-100,
#                 "issues": ["..."],
#                 "suggestions": ["..."]
#             }}

#             User Answer: {answer}
#             """)
#             resp = (eval_prompt | llm).invoke({"answer": message})
#             data = extract_json(resp.content)
#             result.update({
#                 "structure_score": data.get("structure"),
#                 "clarity_score": data.get("clarity"),
#                 "tone_score": data.get("tone"),
#                 "relevance_score": data.get("relevance"),
#                 "detected_issues": data.get("issues", []),
#                 "suggestions": data.get("suggestions", [])
#             })
#         elif looks_like_code(message):
#             code_prompt = PromptTemplate.from_template("""
#             You are a senior software engineer. Analyze the code and return ONLY this JSON:

#             {{
#                 "logic_score": 0-100,
#                 "code_clarity": 0-100,
#                 "efficiency_score": 0-100,
#                 "best_practices": 0-100,
#                 "issues": ["..."],
#                 "suggestions": ["..."]
#             }}

#             Code:\n{code}
#             """)
#             resp = (code_prompt | llm).invoke({"code": message})
#             data = extract_json(resp.content)
#             result.update({
#                 "logic_score": data.get("logic_score"),
#                 "code_clarity": data.get("code_clarity"),
#                 "efficiency_score": data.get("efficiency_score"),
#                 "best_practices": data.get("best_practices"),
#                 "detected_issues": data.get("issues", []),
#                 "suggestions": data.get("suggestions", [])
#             })
#     except Exception as e:
#         result["suggestions"].append("Couldn't evaluate input. Try rephrasing or simplifying.")

#     return ChatResponse(
#         reply=reply_text,
#         total_questions=user_logs[user_id]["questions"],
#         question_types=dict(user_logs[user_id]["types"]),
#         **result
#     )

#GROQ_API_KEY=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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union
from dotenv import load_dotenv
import os
import json
import re
from collections import defaultdict
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Setup OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# FastAPI app init
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace this with your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory user tracking
user_logs = defaultdict(lambda: {"questions": 0, "types": defaultdict(int)})

# Request & Response models
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

# Helper functions
def classify_question(msg: str) -> str:
    msg = msg.lower()
    if any(x in msg for x in ["tree", "graph", "leetcode", "array", "function", "runtime", "code"]):
        return "Coding"
    elif "interview" in msg or "yourself" in msg:
        return "HR"
    else:
        return "General"

def looks_like_answer(msg: str) -> bool:
    return len(msg.split()) > 10 and any(w in msg.lower() for w in ["i ", "my ", "project", "experience", "we "])

def looks_like_code(msg: str) -> bool:
    return any(w in msg.lower() for w in ["def ", "class ", "#include", "public static", "```", "{"])

def extract_json(text: str) -> dict:
    try:
        json_str = re.search(r"{.*}", text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {}

# Chat Endpoint
@app.post("/agent-chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest):
    message = request.message.strip()
    user_id = request.user_id

    if not message:
        message = "Generate a random interview question from DSA, HR, or general topics."

    qtype = classify_question(message)
    user_logs[user_id]["questions"] += 1
    user_logs[user_id]["types"][qtype] += 1

    # Initial reply
    prompt = f"""
You are an AI Interview Coach helping users with HR, coding, and general interview prep.
- If they ask a question, answer clearly.
- If it's their answer, evaluate and give suggestions.
- If it's code, review it and provide feedback.
Respond in helpful, clean Markdown.

User: {message}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",  # replace in prod
                "X-Title": "AI-Interview-Coach"
            }
        )
        reply_text = response.choices[0].message.content
    except Exception as e:
        return ChatResponse(
            reply=f"Error: {str(e)}",
            total_questions=user_logs[user_id]["questions"],
            question_types=dict(user_logs[user_id]["types"]),
        )

    # Evaluate if needed
    result = {
        "structure_score": None, "clarity_score": None, "tone_score": None,
        "relevance_score": None, "logic_score": None, "code_clarity": None,
        "efficiency_score": None, "best_practices": None,
        "detected_issues": [], "suggestions": []
    }

    try:
        if looks_like_answer(message):
            eval_prompt = f"""
You are an interview coach. Evaluate the candidate's answer. Give only this JSON format:

{{
  "structure": 0-100,
  "clarity": 0-100,
  "tone": 0-100,
  "relevance": 0-100,
  "issues": ["..."],
  "suggestions": ["..."]
}}

User Answer: {message}
"""
            eval_resp = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": eval_prompt}]
            )
            data = extract_json(eval_resp.choices[0].message.content)
            result.update({
                "structure_score": data.get("structure"),
                "clarity_score": data.get("clarity"),
                "tone_score": data.get("tone"),
                "relevance_score": data.get("relevance"),
                "detected_issues": data.get("issues", []),
                "suggestions": data.get("suggestions", [])
            })

        elif looks_like_code(message):
            code_prompt = f"""
You are a senior engineer. Review the code and respond in this JSON format only:

{{
  "logic_score": 0-100,
  "code_clarity": 0-100,
  "efficiency_score": 0-100,
  "best_practices": 0-100,
  "issues": ["..."],
  "suggestions": ["..."]
}}

Code:\n{message}
"""
            eval_resp = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": code_prompt}]
            )
            data = extract_json(eval_resp.choices[0].message.content)
            result.update({
                "logic_score": data.get("logic_score"),
                "code_clarity": data.get("code_clarity"),
                "efficiency_score": data.get("efficiency_score"),
                "best_practices": data.get("best_practices"),
                "detected_issues": data.get("issues", []),
                "suggestions": data.get("suggestions", [])
            })
    except Exception as e:
        result["suggestions"].append("Could not evaluate response due to error.")

    return ChatResponse(
        reply=reply_text,
        total_questions=user_logs[user_id]["questions"],
        question_types=dict(user_logs[user_id]["types"]),
        **result
    )
