import os
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from prototype import build_rag_pipeline

load_dotenv()

class TutorState(TypedDict):
    student_id: str
    topic: str
    student_question: str
    explanation_text: str
    student_answer: str
    quiz_data: str
    evaluation_result: dict
    remedial_content: str
    final_report: dict
    profile_metrics: dict
    next_step: str