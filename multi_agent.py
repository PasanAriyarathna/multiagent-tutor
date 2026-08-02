import os
import json
import time
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

router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=800)
quiz_eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=800)
report_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=800)

retriever = build_rag_pipeline()

def invoke_with_retry(llm, prompt, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate_limit" in err_msg or "429" in err_msg:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(2 ** retries)
            else:
                raise e

# --- 3. Agent Nodes ---

def router_agent(state: TutorState) -> TutorState:
    """Agent 0: Router Pattern."""
    current_step = state.get("next_step")
    
    if current_step == "end":
        return state
        
    if current_step in ["explain", "generate_quiz", "evaluate_answer", "generate_report"]:
        return state

    # Fallback / state machine logic for evaluation loop
    if state.get("evaluation_result") and state.get("evaluation_result").get("weak_concepts"):
        state["next_step"] = "remedial_rag"
    else:
        state["next_step"] = "end"
    return state

def explainer_agent(state: TutorState) -> TutorState:
    """Agent 1: Tool-use Pattern for explaining lessons."""
    context = ""
    if retriever:
        docs = retriever.invoke(f"{state['topic']} {state['student_question']}")
        context = "\n".join([d.page_content[:500] for d in docs])

    prompt = f"Using context:\n{context}\n\nExplain '{state['student_question']}' simply for a beginner."
    response = invoke_with_retry(router_llm, prompt)
    state["explanation_text"] = response.content
    state["next_step"] = "end"
    return state

def quiz_generation_agent(state: TutorState) -> TutorState:
    """Agent 2: Generates structured quiz from RAG context."""
    context = ""
    if retriever:
        docs = retriever.invoke(state['topic'])
        context = "\n".join([d.page_content[:500] for d in docs])

    prompt = f"""Context:\n{context}\n\nGenerate a JSON quiz on {state['topic']}. Include at least 1 code snippet question.
    Generate questions ONLY using facts, definitions, and examples found in the provided context below. Do NOT introduce any information that is not present in the context. If the context does not contain enough material for a question type, skip it rather than inventing content.
    Format MUST be exactly: {{"questions": [{{"type": "mcq|code|short_answer", "question": "...", "options": [], "answer": "...", "source_context": "..."}}]}}"""

    response = invoke_with_retry(quiz_eval_llm, prompt)
    state["quiz_data"] = response.content
    return state

def evaluation_agent(state: TutorState) -> TutorState:
    """Agent 3: Reflection/Self-Critique Pattern (Code-aware)."""
    context = ""
    if retriever:
        docs = retriever.invoke(state['topic'])
        context = "\n".join([d.page_content[:500] for d in docs])

    prompt = f"""Context: {context}
    Student Answer Data: {state['student_answer']}
    Evaluate the answer strictly based on the context. Compare the student's submitted answer against the correct answer for this exact question AND the original source_context.
    Instruct the LLM clearly: Compare the student_answer to the correct_answer for this exact question. If student_answer is empty, whitespace, or clearly does not attempt the question, verdict must be 'incorrect'. Mark the answer as 'correct' if the student's answer contains or clearly conveys the correct answer's core meaning, even if phrased differently, more verbosely, or with additional correct supporting detail. Only mark 'incorrect' if the core answer is factually wrong, missing, or contradicts the correct_answer. Do not penalize a student for adding correct extra explanation alongside the right answer.
    Produce a clear per-question correctness verdict: exactly "correct" or "incorrect".
    - If verdict is "correct": do NOT generate any additional explanation. leave weak_concepts empty.
    - If verdict is "incorrect": write a thorough, beginner-friendly explanation (4-8 sentences) scoped ONLY to this specific question's concept. Your explanation MUST: (1) explain why the given answer is wrong (what misconception it reflects), (2) explain why the correct answer is right using the provided source_context, (3) include a concrete example or analogy from the context if available, and (4) end with one clear key takeaway sentence. If no answer was submitted, still provide this full educational explanation — just note briefly at the start that no answer was given, then proceed to teach the concept fully.
    Return strictly JSON:
    {{"question": "...", "student_answer": "...", "correct_answer": "...", "verdict": "correct" | "incorrect", "explanation": "...", "weak_concepts": ["..."]}}"""

    response = invoke_with_retry(quiz_eval_llm, prompt)

    try:
        json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
        eval_dict = json.loads(json_str)
    except:
        eval_dict = {"verdict": "incorrect", "weak_concepts": ["unknown"], "explanation": "Failed to parse JSON."}

    state["evaluation_result"] = eval_dict

    metrics = state.get("profile_metrics", {})
    topic_metrics = metrics.get(state["topic"], {})
    for wc in eval_dict.get("weak_concepts", []):
        topic_metrics[wc] = 0
    for sc in eval_dict.get("strong_concepts", []):
        topic_metrics[sc] = 100
    metrics[state["topic"]] = topic_metrics
    
    # Store QA history for report generation
    qa_history = metrics.get("qa_history", [])
    qa_history.append({
        "student_answer_data": state.get("student_answer", ""),
        "verdict": eval_dict.get("verdict", "incorrect"),
        "explanation": eval_dict.get("explanation", "")
    })
    metrics["qa_history"] = qa_history
    state["profile_metrics"] = metrics

    state["next_step"] = ""
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
                context = "\n".join([d.page_content[:500] for d in docs])
            prompt = f"""Using context:
{context}

The student answered incorrectly.
{state.get('student_answer')}

CRITICAL RULE: Your explanation MUST be 100% specific to this exact question and the correct answer provided.
- STRICTLY PROHIBIT falling back to general topic re-explanations (e.g., general definitions of the main topic or unrelated concepts).
- Explain ONLY why the correct answer is correct for this specific problem, and why the student's answer is wrong.
- Keep the feedback concise and targeted."""
            remedial_texts.append(invoke_with_retry(report_llm, prompt).content)
        state["remedial_content"] = "\n\n---\n\n".join(remedial_texts)

    state["evaluation_result"]["weak_concepts"] = []
    state["next_step"] = "end"
    return state

def report_generation_agent(state: TutorState) -> TutorState:
    """Agent 5: Generates structured JSON report."""
    metrics = state.get("profile_metrics", {})
    qa_history = metrics.get("qa_history", [])
    history_str = json.dumps(qa_history)
    
    prompt = f"""You are an expert AI tutor. Review the following student performance history for this session:
{history_str}

Review each question's verdict and explanation. 
Identify patterns across the incorrect answers (e.g., recurring misconceptions, a specific sub-topic that's consistently wrong). 
Identify what the correct answers reveal about the student's actual strengths (not just that they got it right, but what concept they clearly understand well). 
Write the 'strengths' and 'weak_areas' fields based on this real analysis of the specific questions and answers, not generic statements.

Return strictly JSON format:
{{"strengths": ["..."], "weak_areas": ["..."], "recommended_lessons": ["lesson.pdf"], "revision_order": ["..."], "next_topics": ["..."]}}"""

    response = invoke_with_retry(report_llm, prompt)
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
        "generate_report": "report_agent",
        "end": END
    })

    workflow.add_edge("explainer", "router")
    workflow.add_edge("quiz_agent", END)
    workflow.add_edge("eval_agent", "router")
    workflow.add_edge("remedial_agent", "router")
    workflow.add_edge("report_agent", END)

    return workflow.compile()
