import streamlit as st
from multi_agent import build_tutor_graph, TutorState
from topic_extractor import extract_topics_from_lessons

st.set_page_config(page_title="Intelligent Tutor", layout="wide")

graph = build_tutor_graph()

def safe_invoke(state):
    try:
        return graph.invoke(state)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e).lower():
            st.error("The tutor is briefly busy — please wait a few seconds and try again.")
            if st.button("Retry"):
                st.rerun()
            st.stop()
        raise e

if "state" not in st.session_state:
    st.session_state.state = TutorState(
        student_id="S001", topic="Python Basics", student_question="",
        explanation_text="", student_answer="", quiz_data="",
        evaluation_result={}, remedial_content="", final_report={},
        profile_metrics={}, next_step=""
    )

st.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 5px;">
    <div style="background-color: var(--primary-color, #4F46E5); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        🧠
    </div>
    <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700;">AI Tutoring & Adaptive RAG System</h1>
</div>
<p style="color: #9CA3AF; margin-bottom: 2rem; font-size: 1.05rem; margin-top: 0;">Your intelligent, adaptive learning companion.</p>
""", unsafe_allow_html=True)

@st.cache_data
def load_topics():
    return extract_topics_from_lessons()

with st.spinner("Loading available topics..."):
    categories = load_topics()

st.markdown("""
<style>
.card-container {
    background-color: var(--secondary-background-color, #1E1E2E);
    border-radius: 16px;
    border: 0.5px solid rgba(255,255,255,0.1);
    padding: 1.25rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

if not categories:
    st.warning("Could not detect topics automatically — please type a topic manually.")
    topic = st.text_input("What topic are you studying?", value="Python Loops")
else:
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("Choose a category:", list(categories.keys()))
    with col2:
        topic = st.selectbox("Choose a topic:", categories[selected_category])

question = st.text_input("Your Question:", key="question_input")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    ask_clicked = st.button("Ask a question about this topic", use_container_width=True)
with col_btn2:
    quiz_clicked = st.button("Take a Quiz", type="primary", use_container_width=True)

if ask_clicked:
    if question:
        st.session_state.state["topic"] = topic
        st.session_state.state["student_question"] = question
        st.session_state.state["next_step"] = "explain"
        with st.spinner("Explaining..."):
            st.session_state.state = safe_invoke(st.session_state.state)
        st.info(st.session_state.state["explanation_text"])

if quiz_clicked:
    with st.spinner("Generating Quiz..."):
        st.session_state.state["topic"] = topic
        st.caption(f"Debug: generating quiz for topic = {st.session_state.state['topic']}")
        st.session_state.state["quiz_data"] = ""
        st.session_state.state["evaluation_result"] = {}
        st.session_state.state["next_step"] = "generate_quiz"
        st.session_state.state = safe_invoke(st.session_state.state)
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
            
        metrics = st.session_state.state.get("profile_metrics", {})
        history = metrics.get("qa_history", [])
        correct_count = sum(1 for item in history if item.get("verdict") == "correct")
        incorrect_count = sum(1 for item in history if item.get("verdict") == "incorrect")
        
        col_q, col_score = st.columns([3, 1])
        with col_q:
            st.markdown(f"### Question {idx+1}: {q_type_str}")
        with col_score:
            if history:
                st.markdown(f"<div style='text-align: right; color: #9CA3AF; font-size: 14px; margin-top: 20px;'>✅ Correct: {correct_count} &nbsp;|&nbsp; ❌ Incorrect: {incorrect_count}</div>", unsafe_allow_html=True)

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
                
                # Clear previous evaluation states before invoking graph
                st.session_state.state["evaluation_result"] = {}
                st.session_state.state["remedial_content"] = ""
                
                st.session_state.state["next_step"] = "evaluate_answer"
                with st.spinner("Checking your answer..."):
                    st.session_state.state = safe_invoke(st.session_state.state)
                    st.session_state.question_state = "evaluated"
                    st.rerun()

        elif q_state == "evaluated":
            eval_res = st.session_state.state.get("evaluation_result", {})
            verdict = eval_res.get('verdict', 'incorrect').lower()
            
            if verdict == "correct":
                st.markdown("""
<div style="background-color: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); border-radius: 14px; padding: 1.1rem 1.25rem; margin-bottom: 1.5rem;">
    <div style="color: #22C55E; font-weight: bold; font-size: 1.1rem; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 1.3rem;">✅</span> Correct
    </div>
    <div style="color: var(--text-color, #E5E7EB); font-size: 14px;">
        Great job! You nailed it. Keep up the good work.
    </div>
</div>
""", unsafe_allow_html=True)
            else:
                # Extract student's submitted answer
                student_ans_str = st.session_state.state.get("student_answer", "")
                my_ans = student_ans_str.split("My Answer:")[-1].strip() if "My Answer:" in student_ans_str else "No answer submitted"
                
                correct_ans = q.get('answer', '')
                
                # Pick ONE clear source for the explanation
                explanation = st.session_state.state.get("remedial_content")
                if not explanation or not str(explanation).strip():
                    explanation = eval_res.get('explanation', "No explanation provided.")
                
                # Strip raw markdown headers so it renders cleanly in the error box
                explanation = str(explanation).replace("### ", "**").replace("## ", "**").replace("# ", "**")
                
                if not my_ans or my_ans.lower() == "none" or my_ans == "":
                    my_ans = "No answer submitted"

                # Single, clean colored box display
                import html
                my_ans_safe = html.escape(my_ans)
                correct_ans_safe = html.escape(str(correct_ans))
                
                st.markdown(f"""
<div style="background-color: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); border-radius: 14px; padding: 1.1rem 1.25rem; margin-bottom: 1.5rem;">
    <div style="color: #EF4444; font-weight: bold; font-size: 1.1rem; display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
        <span style="font-size: 1.3rem;">❌</span> Incorrect
    </div>
    <div style="display: grid; grid-template-columns: 120px 1fr; gap: 8px 16px; margin-bottom: 20px; font-size: 14px;">
        <div style="color: #9CA3AF;">Your answer:</div>
        <div style="color: var(--text-color, #F3F4F6); font-weight: 500;">{my_ans_safe}</div>
        <div style="color: #9CA3AF;">Correct answer:</div>
        <div style="color: #22C55E; font-weight: 500;">{correct_ans_safe}</div>
    </div>
    <div style="line-height: 1.6; font-size: 14px; color: var(--text-color, #E5E7EB); border-top: 1px solid rgba(255,255,255,0.1); padding-top: 16px;">

{explanation}

    </div>
</div>
""", unsafe_allow_html=True)
                print(f"DEBUG [Question {idx+1}]: Rendering explanation -> {explanation[:100]}...")
                
            st.markdown("Are you ready for the next question?")
            if st.button("Next Question"):
                st.session_state.current_question_index += 1
                st.session_state.question_state = "waiting_for_answer"
                st.session_state.state["evaluation_result"] = {}
                st.session_state.state["remedial_content"] = ""
                st.session_state.state["student_answer"] = ""
                st.rerun()

    elif questions and idx >= len(questions):
        st.success("You have completed all questions in this quiz!")
        if st.button("Get Final Progress Report"):
            st.session_state.state["next_step"] = "generate_report"
            with st.spinner("Generating Report..."):
                st.session_state.state = safe_invoke(st.session_state.state)
            
            report = st.session_state.state.get("final_report", {})
            st.markdown("## 🎓 Final Progress Report")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 💪 Strengths")
                for s in report.get("strengths", []):
                    st.markdown(f"- {s}")
            with col2:
                st.markdown("### ⚠️ Areas for Improvement")
                for w in report.get("weak_areas", []):
                    st.markdown(f"- {w}")
            
            st.markdown("### 📚 Recommended Revision Order")
            for i, rev in enumerate(report.get("revision_order", [])):
                st.markdown(f"{i+1}. {rev}")
                
            st.markdown("### ⏭️ Next Topics")
            for t in report.get("next_topics", []):
                st.markdown(f"- {t}")