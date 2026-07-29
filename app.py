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

    if st.button("Submit Answer"):
        st.session_state.state["student_answer"] = answer
        st.session_state.state["next_step"] = "evaluate_answer"
        with st.spinner("Evaluating Code and Logic..."):
            st.session_state.state = st.session_state.graph.invoke(st.session_state.state)

            eval_res = st.session_state.state["evaluation_result"]
            st.write(f"**Score:** {eval_res.get('score')}")
            st.write(f"**Feedback:** {eval_res.get('feedback')}")
            if eval_res.get("code_errors"):
                st.error(eval_res["code_errors"])

            if st.session_state.state.get("remedial_content"):
                st.warning("### Remedial Practice\n" + st.session_state.state["remedial_content"])

            if st.button("Get Final Progress Report"):
                st.session_state.state["next_step"] = "generate_report"
                st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
                st.json(st.session_state.state["final_report"])