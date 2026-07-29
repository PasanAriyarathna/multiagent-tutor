import streamlit as st
from multi_agent import build_tutor_graph, TutorState

st.set_page_config(page_title="Intelligent Tutor", layout="wide")

if "graph" not in st.session_state:
    st.session_state.graph = build_tutor_graph()
if "state" not in st.session_state:
    st.session_state.state = TutorState(
        student_id="S001", topic="Python Basics", student_question="",
        explanation_text="", student_answer="", quiz_data="",
        evaluation_result={}, remedial_content="", final_report={},
        profile_metrics={}, next_step=""
    )

st.title("🧠 AI Tutoring & Adaptive RAG System")