from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # ✅ Use Groq instead of ChatOpenAI
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Dummy in-memory log (for production, use a real DB)
user_logs = defaultdict(lambda: {"questions": 0, "types": defaultdict(int)})

# Chat model (Groq)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",  # 🔁 You can switch to mixtral or gemma
    temperature=0.7
)

# Request & Response models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    total_questions: int
    question_types: dict

# Question classifier
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

# Endpoint
@app.post("/agent-chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest):
    message = request.message.strip()
    user_id = request.user_id

    # Update tracking
    qtype = classify_question(message)
    user_logs[user_id]["questions"] += 1
    user_logs[user_id]["types"][qtype] += 1

    # Prompt
    prompt = PromptTemplate.from_template(
        """
        You are an AI Interview Coach. Understand this user message and respond clearly.

        - If it's a question, answer it like an expert.
        - If it's an answer attempt by the user, give feedback.
        - Keep it brief, helpful, and formatted in Markdown.

        User: {prompt}
        """
    )

    # Invoke model
    response = (prompt | llm).invoke({"prompt": message})

    # Respond with data
    return ChatResponse(
        reply=response.content,
        total_questions=user_logs[user_id]["questions"],
        question_types=dict(user_logs[user_id]["types"])
    )
