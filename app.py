import json
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from generator_groq import generate_answer
from guard_laya import check_guard, get_router
from judge_groq import check_llm_judge
from rag_pipeline import STANDARD_REFUSAL_MESSAGE, run_rag_pipeline
from retriever_pinecone import retrieve

st.set_page_config(
    page_title="RAG Hallucination Guard",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_system_resources():
    router = get_router()
    return router


def load_calibrated_thresholds():
    path = "eval/calibrated_thresholds.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "deliver_threshold": 0.80,
        "retry_threshold": 0.50,
        "context_sufficient_threshold": 0.50
    }


def render_action_badge(action: str):
    if action == "delivered":
        st.markdown(
            '<span style="background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 14px;">✅ Delivered</span>',
            unsafe_allow_html=True
        )
    elif action == "delivered_after_retry":
        st.markdown(
            '<span style="background-color: #17a2b8; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 14px;">🔄 Delivered After Retry</span>',
            unsafe_allow_html=True
        )
    elif "refused" in action:
        st.markdown(
            f'<span style="background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 14px;">⛔ {action.replace("_", " ").title()}</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<span style="background-color: #6c757d; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 14px;">ℹ️ {action.title()}</span>',
            unsafe_allow_html=True
        )


def main():
    load_system_resources()
    calibrated = load_calibrated_thresholds()

    st.title("🛡️ RAG Hallucination Guard")
    st.caption("Active Groundedness & Refusal Guardrail powered by Laya Multilingual Router & Pinecone")

    tab1, tab2 = st.tabs(["🚀 Live Demo", "📊 Evaluation Results"])

    with tab1:
        st.sidebar.header("⚙️ Guard Settings")
        use_guard = st.sidebar.toggle("Enable Hallucination Guard", value=True)
        side_by_side = st.sidebar.checkbox("Compare Side-by-Side (Guarded vs Baseline)", value=True)

        deliver_th = st.sidebar.slider(
            "Deliver Threshold (P_supported)",
            min_value=0.30,
            max_value=0.95,
            value=float(calibrated.get("deliver_threshold", 0.80)),
            step=0.05,
            help="Minimum groundedness score required to deliver an answer."
        )

        retry_th = st.sidebar.slider(
            "Retry Threshold (P_supported)",
            min_value=0.20,
            max_value=float(deliver_th),
            value=float(min(calibrated.get("retry_threshold", 0.50), deliver_th)),
            step=0.05,
            help="Middle-band threshold: answers between retry and deliver thresholds trigger a retry with k=8 passages."
        )

        context_th = st.sidebar.slider(
            "Context Sufficiency Threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(calibrated.get("context_sufficient_threshold", 0.50)),
            step=0.05,
            help="If context sufficiency score falls below this, the query is refused immediately."
        )

        st.subheader("Try Example Questions")
        example_cols = st.columns(2)
        with example_cols[0]:
            st.markdown("**Answerable in SQuAD Corpus:**")
            ex1 = st.button("🏰 Who conquered England killing Harold II at Hastings?")
            ex2 = st.button("🇫🇷 In what country is Normandy located?")
            ex3 = st.button("📜 When was the Latin version of Norman first recorded?")

        with example_cols[1]:
            st.markdown("**Unanswerable / Distractors:**")
            ex4 = st.button("❓ What is France a region of?")
            ex5 = st.button("👑 Who did King Charles III swear fealty to?")
            ex6 = st.button("🏰 What principality did William the Conqueror found?")

        selected_query = ""
        if ex1:
            selected_query = "Who conquered England killing King Harold II at the Battle of Hastings?"
        elif ex2:
            selected_query = "In what country is Normandy located?"
        elif ex3:
            selected_query = "When was the Latin version of the word Norman first recorded?"
        elif ex4:
            selected_query = "What is France a region of?"
        elif ex5:
            selected_query = "Who did King Charles III swear fealty to?"
        elif ex6:
            selected_query = "What principality did William the conquerer found?"

        query = st.text_input(
            "Enter Question:",
            value=selected_query or st.session_state.get("current_query", "Who conquered England killing King Harold II at the Battle of Hastings?")
        )
        st.session_state["current_query"] = query

        ask_btn = st.button("🔍 Ask Pipeline", type="primary")

        if ask_btn and query:
            with st.spinner("Retrieving passages, generating answer, and verifying with Laya guard..."):
                try:
                    t_start = time.perf_counter()
                    raw_passages = retrieve(query, k=4)
                    raw_draft = generate_answer(query, raw_passages, prompt_variant="strict")

                    guarded_res = run_rag_pipeline(
                        question=query,
                        use_guard=use_guard,
                        guard_type="laya",
                        deliver_threshold=deliver_th,
                        retry_threshold=retry_th,
                        context_sufficient_threshold=context_th
                    )

                    baseline_res = None
                    if side_by_side:
                        baseline_res = run_rag_pipeline(
                            question=query,
                            use_guard=False
                        )

                    st.markdown("---")

                    if side_by_side and baseline_res:
                        col_g, col_b = st.columns(2)

                        with col_g:
                            st.markdown("### 🛡️ Guarded Pipeline")
                            render_action_badge(guarded_res["action"])
                            st.markdown(f"**Final Answer:**\n\n> {guarded_res['answer']}")

                            if guarded_res["scores"]:
                                p_sup = guarded_res["scores"]["p_supported"]
                                p_suf = guarded_res["scores"]["p_context_sufficient"]
                                m_col1, m_col2 = st.columns(2)
                                m_col1.metric("Grounded Score (P_sup)", f"{p_sup:.4f}")
                                m_col2.metric("Sufficiency Score (P_suf)", f"{p_suf:.4f}")
                                st.progress(float(p_sup), text=f"Groundedness: {int(p_sup*100)}%")
                                st.progress(float(p_suf), text=f"Context Sufficiency: {int(p_suf*100)}%")

                            st.caption(f"⏱️ Response Time: {guarded_res['time_taken']:.2f}s")

                        with col_b:
                            st.markdown("### ⚡ Unguarded Baseline")
                            render_action_badge(baseline_res["action"])
                            st.markdown(f"**Final Answer:**\n\n> {baseline_res['answer']}")
                            st.info("No verification applied. Model output directly delivered.")
                            st.caption(f"⏱️ Response Time: {baseline_res['time_taken']:.2f}s")
                    else:
                        st.markdown("### Output")
                        render_action_badge(guarded_res["action"])
                        st.markdown(f"**Final Answer:**\n\n> {guarded_res['answer']}")

                        if guarded_res["scores"]:
                            p_sup = guarded_res["scores"]["p_supported"]
                            p_suf = guarded_res["scores"]["p_context_sufficient"]
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Grounded Score", f"{p_sup:.4f}")
                            m_col2.metric("Sufficiency Score", f"{p_suf:.4f}")
                            st.progress(float(p_sup), text=f"Groundedness: {int(p_sup*100)}%")
                            st.progress(float(p_suf), text=f"Context Sufficiency: {int(p_suf*100)}%")

                        st.caption(f"⏱️ Response Time: {guarded_res['time_taken']:.2f}s")

                    with st.expander("📝 Raw Drafted Answer (Before Guard Check)"):
                        st.write(raw_draft)

                    with st.expander("📚 Retrieved Passages"):
                        passages_to_show = guarded_res.get("passages", raw_passages)
                        for i, p in enumerate(passages_to_show, 1):
                            st.markdown(f"**Passage [{i}]:**\n{p}")

                except Exception as exc:
                    st.error(f"Error during execution: {exc}")

    with tab2:
        st.header("📊 Guard Evaluation & Calibration Analysis")

        benchmark_summary_path = "eval/benchmark_summary.csv"
        sweep_path = "eval/threshold_sweep.csv"
        raw_scores_path = "eval/tuning_raw_scores.json"
        benchmark_results_path = "eval/benchmark_results.json"

        st.subheader("1. Pipeline Comparison Table")
        if os.path.exists(benchmark_summary_path):
            df_summary = pd.read_csv(benchmark_summary_path)
            st.dataframe(
                df_summary[[
                    "run_key", "config", "prompt_variant", "total_queries",
                    "wrong_answers_delivered", "wrong_answers_delivered_pct",
                    "correct_answers_wrongly_refused", "correct_answers_wrongly_refused_pct",
                    "avg_latency_sec"
                ]],
                use_container_width=True
            )
        else:
            st.info("Benchmark summary file not found yet. Run `python eval/evaluate_system.py` to generate.")

        st.markdown("---")
        st.subheader("2. Threshold Trade-Off Curve (Precision vs Coverage)")
        if os.path.exists(sweep_path):
            df_sweep = pd.read_csv(sweep_path)
            fig_sweep = go.Figure()

            fig_sweep.add_trace(go.Scatter(
                x=df_sweep["threshold"],
                y=df_sweep["unanswerable_refused_pct"],
                mode="lines+markers",
                name="Unanswerable Refused % (Safety)",
                line=dict(color="#28a745", width=3)
            ))

            fig_sweep.add_trace(go.Scatter(
                x=df_sweep["threshold"],
                y=df_sweep["answerable_delivered_pct"],
                mode="lines+markers",
                name="Answerable Delivered % (Recall)",
                line=dict(color="#007bff", width=3)
            ))

            chosen_th = float(calibrated.get("deliver_threshold", 0.80))
            fig_sweep.add_vline(
                x=chosen_th,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Selected Threshold ({chosen_th})",
                annotation_position="top left"
            )

            fig_sweep.update_layout(
                title="Sweep Trade-off: Unanswerable Refusal vs Answerable Delivery",
                xaxis_title="Threshold (P_supported)",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig_sweep, use_container_width=True)
        else:
            st.info("Sweep data not found. Run threshold sweep to plot.")

        st.markdown("---")
        st.subheader("3. Grounded Score Separation: Answerable vs Unanswerable")
        if os.path.exists(raw_scores_path):
            with open(raw_scores_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            df_raw = pd.DataFrame(raw_data)
            df_raw["Group"] = df_raw["is_answerable"].map({True: "Answerable", False: "Unanswerable"})

            fig_hist = px.histogram(
                df_raw,
                x="p_supported",
                color="Group",
                barmode="overlay",
                nbins=20,
                color_discrete_map={"Answerable": "#007bff", "Unanswerable": "#dc3545"},
                opacity=0.7,
                title="Separation of Grounded Scores (P_supported) by Question Type"
            )
            fig_hist.update_layout(
                xaxis_title="Grounded Score (P_supported)",
                yaxis_title="Count",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Tuning raw scores not found. Run tuning to plot distribution.")

        st.markdown("---")
        st.subheader("4. Error Analysis: Failure Cases")
        if os.path.exists(benchmark_results_path):
            with open(benchmark_results_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)

            failures = []
            for run_k, run_val in b_data.items():
                for d in run_val.get("details", []):
                    if d.get("is_wrong_delivered"):
                        failures.append({
                            "Configuration": run_k,
                            "Failure Type": "Wrongly Delivered (Hallucination)",
                            "Question": d["question"],
                            "Answer Delivered": d["delivered_answer"],
                            "Action": d["action"],
                            "P_supported": d["scores"]["p_supported"] if d.get("scores") else None,
                            "P_sufficient": d["scores"]["p_context_sufficient"] if d.get("scores") else None
                        })
                    elif d.get("is_correct_refused"):
                        failures.append({
                            "Configuration": run_k,
                            "Failure Type": "Wrongly Refused (False Rejection)",
                            "Question": d["question"],
                            "Answer Delivered": d["delivered_answer"],
                            "Action": d["action"],
                            "P_supported": d["scores"]["p_supported"] if d.get("scores") else None,
                            "P_sufficient": d["scores"]["p_context_sufficient"] if d.get("scores") else None
                        })

            if failures:
                df_failures = pd.DataFrame(failures)
                st.dataframe(df_failures, use_container_width=True)
            else:
                st.success("No failure cases recorded in benchmark.")
        else:
            st.info("Benchmark details not found. Run benchmark to inspect failure cases.")


if __name__ == "__main__":
    main()
