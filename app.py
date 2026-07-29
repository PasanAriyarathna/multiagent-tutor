import streamlit as st
from multi_agent import build_tutor_graph, TutorState

st.set_page_config(page_title="Intelligent Tutor", layout="wide")

graph = build_tutor_graph()
if "state" not in st.session_state:
    st.session_state.state = TutorState(
        student_id="S001", topic="Python Basics", student_question="",
        explanation_text="", student_answer="", quiz_data="",
        evaluation_result={}, remedial_content="", final_report={},
        profile_metrics={}, next_step=""
    )

st.title("🧠 AI Tutoring & Adaptive RAG System")

topic = st.text_input("What topic are you studying?", value="Python Loops")

question = st.text_input("Your Question:", key="question_input")
if st.button("Ask a question about this topic"):
    if question:
        st.session_state.state["topic"] = topic
        st.session_state.state["student_question"] = question
        st.session_state.state["next_step"] = "explain"
        with st.spinner("Explaining..."):
            st.session_state.state = graph.invoke(st.session_state.state)
        st.info(st.session_state.state["explanation_text"])

if st.button("Take a Quiz"):
    with st.spinner("Generating Quiz..."):
        st.session_state.state["topic"] = topic
        st.session_state.state["next_step"] = "generate_quiz"
        st.session_state.state = graph.invoke(st.session_state.state)
        st.session_state.pop("quiz_questions", None)
        st.session_state.pop("current_question_index", None)
        st.session_state.pop("question_state", None)
        st.success("Quiz Generated!")

if st.session_state.state.get("quiz_data"):
    if "quiz_questions" not in st.session_state:
        try:
            import json
            quiz_json_str = st.session_state.state["quiz_data"]
            start_idx = quiz_json_str.find('{')
            end_idx = quiz_json_str.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                quiz_dict = json.loads(quiz_json_str[start_idx:end_idx])
            else:
                quiz_dict = json.loads(quiz_json_str)
            st.session_state.quiz_questions = quiz_dict.get("questions", [])
            st.session_state.current_question_index = 0
            st.session_state.question_state = "waiting_for_answer"
        except Exception as e:
            st.session_state.quiz_questions = []

    questions = st.session_state.get("quiz_questions", [])
    idx = st.session_state.get("current_question_index", 0)
    q_state = st.session_state.get("question_state", "waiting_for_answer")

    if questions and idx < len(questions):
        q = questions[idx]
        q_type = q.get("type", "General")
        if q_type.lower() == "mcq":
            q_type_str = "Multiple Choice"
        elif q_type.lower() == "short_answer":
            q_type_str = "Short Answer"
        elif q_type.lower() == "code":
            q_type_str = "Code"
        else:
            q_type_str = q_type.replace("_", " ").title()
            
        st.markdown(f"### Question {idx+1}: {q_type_str}")
        st.markdown(f"**Question:** {q.get('question', '')}")
        
        options = q.get("options", [])
        is_mcq = q_type.lower() == "mcq" and isinstance(options, list) and len(options) > 0
        
        if not is_mcq and options and isinstance(options, list) and len(options) > 0:
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
            for j, opt in enumerate(options):
                label = labels[j] if j < len(labels) else str(j+1)
                st.markdown(f"* {label}) {opt}")
                
        if q_state == "waiting_for_answer":
            if is_mcq:
                answer = st.radio("Select your answer:", options, key=f"ans_radio_{idx}")
            else:
                st.markdown("*(Wait for user input before revealing the answer)*")
                answer = st.text_area("Your Answer:", key=f"ans_{idx}")
                
            if st.button("Submit Answer"):
                st.session_state.state["student_answer"] = f"Question: {q.get('question', '')}\nOptions: {q.get('options', [])}\nSource Context: {q.get('source_context', '')}\nCorrect Answer: {q.get('answer', '')}\nMy Answer: {answer}"
                
                if not answer or not str(answer).strip():
                    st.session_state.state["evaluation_result"] = {
                        "verdict": "incorrect_empty",
                        "explanation": "No answer was provided."
                    }
                    st.session_state.question_state = "evaluated"
                    st.rerun()
                else:
                    st.session_state.state["next_step"] = "evaluate_answer"
                    with st.spinner("Evaluating Code and Logic..."):
                        st.session_state.state = graph.invoke(st.session_state.state)
                        st.session_state.question_state = "evaluated"
                        st.rerun()

        elif q_state == "evaluated":
            eval_res = st.session_state.state.get("evaluation_result", {})
            verdict = eval_res.get('verdict', 'incorrect').lower()
            
            if verdict == "correct":
                st.success("✅ Correct")
            elif verdict == "incorrect_empty":
                st.error("❌ Incorrect (no answer submitted)")
                st.info(f"**Correct Answer:** {q.get('answer', '')}")
            else:
                st.error("❌ Incorrect")
                st.info(f"**Correct Answer:** {q.get('answer', '')}")
                
                if eval_res.get('explanation'):
                    st.write(f"**Explanation:** {eval_res.get('explanation')}")
                    
                if st.session_state.state.get("remedial_content"):
                    st.warning("### Remedial Explanation\n" + st.session_state.state["remedial_content"])
                
            st.markdown("Are you ready for the next question?")
            if st.button("Next Question"):
                st.session_state.current_question_index += 1
                st.session_state.question_state = "waiting_for_answer"
                st.rerun()

    elif questions and idx >= len(questions):
        st.success("You have completed all questions in this quiz!")
        if st.button("Get Final Progress Report"):
            st.session_state.state["next_step"] = "generate_report"
            st.session_state.state = graph.invoke(st.session_state.state)
            st.json(st.session_state.state["final_report"])