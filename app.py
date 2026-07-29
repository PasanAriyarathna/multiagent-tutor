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

topic = st.text_input("What topic are you studying?", value="Python Loops")

if st.button("Ask a question about this topic"):
    question = st.text_input("Your Question:")
    if question:
        st.session_state.state["topic"] = topic
        st.session_state.state["student_question"] = question
        st.session_state.state["next_step"] = "explain"
        with st.spinner("Explaining..."):
            st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
        st.info(st.session_state.state["explanation_text"])

if st.button("Take a Quiz"):
    with st.spinner("Generating Quiz..."):
        st.session_state.state["topic"] = topic
        st.session_state.state["next_step"] = "generate_quiz"
        st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
        st.success("Quiz Generated!")

if st.session_state.state.get("quiz_data"):
    st.write(st.session_state.state["quiz_data"])
    answer = st.text_area("Your Answer:")