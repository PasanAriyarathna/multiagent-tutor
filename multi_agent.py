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

router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
quiz_eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
report_llm = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

retriever = build_rag_pipeline()

# --- 3. Agent Nodes ---

def router_agent(state: TutorState) -> TutorState:
    """Agent 0: Router Pattern."""
    if state.get("student_question") and not state.get("explanation_text"):
        state["next_step"] = "explain"
    elif not state.get("quiz_data"):
        state["next_step"] = "generate_quiz"
    elif state.get("student_answer") and not state.get("evaluation_result"):
        state["next_step"] = "evaluate_answer"
    elif state.get("evaluation_result") and state["evaluation_result"].get("weak_concepts"):
        state["next_step"] = "remedial_rag"
    else:
        state["next_step"] = "generate_report"
    return state

def explainer_agent(state: TutorState) -> TutorState:
    """Agent 1: Tool-use Pattern for explaining lessons."""
    context = ""
    if retriever:
        docs = retriever.invoke(f"{state['topic']} {state['student_question']}")
        context = "\n".join([d.page_content for d in docs])

    prompt = f"Using context:\n{context}\n\nExplain '{state['student_question']}' simply for a beginner."
    response = router_llm.invoke(prompt)
    state["explanation_text"] = response.content
    state["next_step"] = "generate_quiz"
    return state
def quiz_generation_agent(state: TutorState) -> TutorState:
    """Agent 2: Generates structured quiz from RAG context."""
    context = ""
    if retriever:
        docs = retriever.invoke(state['topic'])
        context = "\n".join([d.page_content for d in docs])

    prompt = f"""Context:\n{context}\n\nGenerate a JSON quiz on {state['topic']}. Include at least 1 code snippet question.
    Format MUST be exactly: {{"questions": [{{"type": "mcq|code|short_answer", "question": "...", "options": [], "answer": "..."}}]}}"""

    response = quiz_eval_llm.invoke(prompt)
    state["quiz_data"] = response.content
    return state

def evaluation_agent(state: TutorState) -> TutorState:
    """Agent 3: Reflection/Self-Critique Pattern (Code-aware)."""
    context = ""
    if retriever:
        docs = retriever.invoke(state['topic'])
        context = "\n".join([d.page_content for d in docs])

    prompt = f"""Context: {context}
    Student Answer: {state['student_answer']}
    Evaluate the answer strictly based on the context. If there are code errors, classify as syntax_error, logic_error, or concept_mistake.
    Return strictly JSON:
    {{"score": int, "weak_concepts": ["..."], "strong_concepts": ["..."], "code_errors": [{{"type": "logic_error", "description": "...", "correct_code": "...", "explanation": "..."}}], "feedback": "..."}}"""

    response = quiz_eval_llm.invoke(prompt)

    try:
        json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
        eval_dict = json.loads(json_str)
    except:
        eval_dict = {"score": 50, "weak_concepts": ["syntax"], "strong_concepts": [], "code_errors": [], "feedback": "Failed to parse JSON."}

    state["evaluation_result"] = eval_dict

    metrics = state.get("profile_metrics", {})
    topic_metrics = metrics.get(state["topic"], {})
    for wc in eval_dict.get("weak_concepts", []):
        topic_metrics[wc] = 0
    for sc in eval_dict.get("strong_concepts", []):
        topic_metrics[sc] = 100
    metrics[state["topic"]] = topic_metrics
    state["profile_metrics"] = metrics

    return state

def remedial_rag_agent(state: TutorState) -> TutorState:
    """Agent 4: Explains weak concepts and provides targeted practice."""
    weak_concepts = state["evaluation_result"].get("weak_concepts", [])
    if weak_concepts:
        remedial_texts = []
        for wc in weak_concepts:
            context = ""
            if retriever:
                docs = retriever.invoke(f"basic explanation of {wc}")
                context = "\n".join([d.page_content for d in docs])
            prompt = f"Using context:\n{context}\n\nProvide a simplified re-explanation of {wc} with an example, and 2 targeted practice questions."
            remedial_texts.append(report_llm.invoke(prompt).content)
        state["remedial_content"] = "\n\n---\n\n".join(remedial_texts)

    state["evaluation_result"]["weak_concepts"] = []
    return state

def report_generation_agent(state: TutorState) -> TutorState:
    """Agent 5: Generates structured JSON report."""
    metrics_str = json.dumps(state["profile_metrics"])
    prompt = f"""Generate a personalized report based on these metrics: {metrics_str}.
    Return strictly JSON:
    {{"strengths": ["..."], "weak_areas": ["..."], "recommended_lessons": ["lesson.pdf"], "revision_order": ["..."], "next_topics": ["..."]}}"""

    response = report_llm.invoke(prompt)
    try:
        json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
        state["final_report"] = json.loads(json_str)
    except:
        state["final_report"] = {"error": "Could not parse JSON report."}
    return state

# --- 4. Build LangGraph ---
def build_tutor_graph():
    workflow = StateGraph(TutorState)

    workflow.add_node("router", router_agent)
    workflow.add_node("explainer", explainer_agent)
    workflow.add_node("quiz_agent", quiz_generation_agent)
    workflow.add_node("eval_agent", evaluation_agent)
    workflow.add_node("remedial_agent", remedial_rag_agent)
    workflow.add_node("report_agent", report_generation_agent)

    workflow.set_entry_point("router")

    def route_condition(state: TutorState):
        return state["next_step"]

    workflow.add_conditional_edges("router", route_condition, {
        "explain": "explainer",
        "generate_quiz": "quiz_agent",
        "evaluate_answer": "eval_agent",
        "remedial_rag": "remedial_agent",
        "generate_report": "report_agent"
    })

    workflow.add_edge("explainer", "router")
    workflow.add_edge("quiz_agent", END)
    workflow.add_edge("eval_agent", "router")
    workflow.add_edge("remedial_agent", "router")
    workflow.add_edge("report_agent", END)

    return workflow.compile()


