import hashlib
import html
import io
import os
import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MODEL_PATH = "forensic_social_media_model.pkl"


THREAT_TERMS = {
    "attack", "destroy", "shoot", "stab", "bomb", "burn", "hurt", "violence",
    "threat", "kill", "murder", "harm", "dead", "explode"
}

CRIME_TERMS = {
    "stolen", "drugs", "weapon", "gun", "fraud", "scam", "fake id", "hack",
    "password", "credit card", "illegal", "rob", "robbery", "selling stolen"
}

TOXIC_TERMS = {
    "hate", "trash", "useless", "idiot", "stupid", "loser", "worthless",
    "dumb", "scum", "filth", "vermin"
}

TARGET_TERMS = {
    "race", "religion", "gender", "immigrant", "foreigner", "minority",
    "community", "group", "ethnicity"
}


st.set_page_config(
    page_title="COS 783 Social Media Analysis",
    page_icon="DF",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --case-navy: #111820;
                --case-ink: #1d2937;
                --case-border: #c8d2df;
                --case-panel: #f3f6f9;
                --case-accent: #1f7a8c;
                --case-danger: #b42318;
                --case-warn: #b54708;
                --case-ok: #067647;
                --case-lime: #9fd356;
            }

            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            .hero {
                background:
                    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
                    linear-gradient(135deg, #0d131a 0%, #172333 58%, #223a42 100%);
                background-size: 28px 28px, 28px 28px, auto;
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 10px;
                padding: 28px 30px;
                color: #ffffff;
                box-shadow: 0 18px 45px rgba(19, 33, 55, 0.18);
                margin-bottom: 22px;
            }

            .hero h1 {
                margin: 0 0 8px 0;
                font-size: 2.35rem;
                line-height: 1.1;
                color: #ffffff;
            }

            .hero p {
                margin: 0;
                color: #dce7f6;
                max-width: 980px;
                font-size: 1.03rem;
            }

            .badge-row {
                margin-top: 18px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                border-radius: 4px;
                padding: 6px 10px;
                background: rgba(255,255,255,0.11);
                border: 1px solid rgba(255,255,255,0.18);
                color: #f6f9ff;
                font-size: 0.84rem;
            }

            .metric-card {
                border: 1px solid var(--case-border);
                background: #ffffff;
                border-radius: 8px;
                padding: 16px 18px;
                min-height: 112px;
                box-shadow: 0 10px 24px rgba(27, 39, 55, 0.06);
            }

            .metric-card span {
                color: #5c6675;
                font-size: 0.82rem;
                text-transform: uppercase;
                font-weight: 700;
            }

            .metric-card strong {
                color: var(--case-ink);
                display: block;
                font-size: 2rem;
                margin-top: 5px;
            }

            .metric-card small {
                color: #667085;
            }

            .case-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
                margin: 0 0 22px 0;
            }

            .case-tile {
                border: 1px solid var(--case-border);
                background: #ffffff;
                border-radius: 8px;
                padding: 13px 14px;
                box-shadow: 0 8px 20px rgba(27, 39, 55, 0.05);
            }

            .case-tile span {
                display: block;
                color: #667085;
                font-size: 0.75rem;
                text-transform: uppercase;
                font-weight: 800;
            }

            .case-tile strong {
                display: block;
                color: var(--case-ink);
                margin-top: 5px;
                word-break: break-word;
            }

            .tweet-box {
                border: 1px solid var(--case-border);
                border-left: 5px solid var(--case-accent);
                background: #ffffff;
                border-radius: 8px;
                padding: 18px;
                margin-bottom: 14px;
                box-shadow: 0 8px 20px rgba(27, 39, 55, 0.05);
            }

            .tweet-text {
                color: #192232;
                font-size: 1.02rem;
                line-height: 1.55;
                margin-bottom: 13px;
            }

            .evidence {
                background: var(--case-panel);
                border: 1px solid var(--case-border);
                border-radius: 8px;
                padding: 14px 16px;
                margin-top: 12px;
            }

            .custody-panel {
                border: 1px solid var(--case-border);
                background: #ffffff;
                border-radius: 8px;
                padding: 16px;
                min-height: 185px;
                box-shadow: 0 8px 20px rgba(27, 39, 55, 0.05);
            }

            .custody-panel h3 {
                margin-top: 0;
                font-size: 1.05rem;
            }

            .risk-high {
                color: var(--case-danger);
                font-weight: 800;
            }

            .risk-medium {
                color: var(--case-warn);
                font-weight: 800;
            }

            .risk-low {
                color: var(--case-ok);
                font-weight: 800;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--case-border);
                border-radius: 8px;
                padding: 15px 16px;
                box-shadow: 0 8px 20px rgba(27, 39, 55, 0.05);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 8px 8px 0 0;
                padding: 10px 16px;
            }

            @media (max-width: 900px) {
                .case-strip {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 560px) {
                .case-strip {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def redact_text(text):
    text = re.sub(r"@\w+", "@[redacted]", str(text))
    text = re.sub(r"http\S+|www\S+", "[url]", text)
    return text


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", str(text).lower())


sentiment_analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    score = sentiment_analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


def keyword_check(text):
    tokens = set(tokenize(text))
    cleaned = clean_text(text)

    threat_hits = sorted(tokens.intersection(THREAT_TERMS))
    toxic_hits = sorted(tokens.intersection(TOXIC_TERMS))
    target_hits = sorted(tokens.intersection(TARGET_TERMS))

    crime_hits = []
    for term in CRIME_TERMS:
        if term in cleaned:
            crime_hits.append(term)

    if threat_hits:
        rule = "Threat indicator"
    elif crime_hits:
        rule = "Crime/scam indicator"
    elif toxic_hits:
        rule = "Toxic language indicator"
    else:
        rule = "No major rule hit"

    return {
        "rule_detection": rule,
        "threat_terms": ", ".join(threat_hits) if threat_hits else "None",
        "crime_terms": ", ".join(crime_hits) if crime_hits else "None",
        "toxic_terms": ", ".join(toxic_hits) if toxic_hits else "None",
        "target_terms": ", ".join(target_hits) if target_hits else "None"
    }


def calculate_risk(ai_label, rule_detection, sentiment):
    ai_label = str(ai_label).lower()

    if rule_detection == "Threat indicator":
        return "High"
    if ai_label in ["hate", "toxic"] and sentiment == "Negative":
        return "High"
    if ai_label in ["hate", "toxic", "offensive", "misinformation"]:
        return "Medium"
    if rule_detection in ["Crime/scam indicator", "Toxic language indicator"]:
        return "Medium"
    if ai_label == "spam":
        return "Medium"
    if sentiment == "Negative":
        return "Low-Medium"
    return "Low"


def risk_class(risk):
    if risk == "High":
        return "risk-high"
    if risk in ["Medium", "Low-Medium"]:
        return "risk-medium"
    return "risk-low"


def render_metric(label, value, helper):
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
            <small>{html.escape(str(helper))}</small>
        </div>
        """,
        unsafe_allow_html=True
    )


def analyse_text(text, model_bundle):
    cleaned = clean_text(text)
    model = model_bundle["model"]

    ai_label = model.predict([cleaned])[0]

    probability = None
    try:
        probabilities = model.predict_proba([cleaned])[0]
        classes = model.classes_
        class_position = list(classes).index(ai_label)
        probability = float(probabilities[class_position])
    except Exception:
        probability = None

    sentiment, sentiment_score = get_sentiment(text)
    checks = keyword_check(text)
    risk = calculate_risk(ai_label, checks["rule_detection"], sentiment)

    result = {
        "original_text": str(text),
        "cleaned_text": cleaned,
        "redacted_text": redact_text(text),
        "ai_classification": ai_label,
        "ai_confidence": round(probability * 100, 2) if probability is not None else None,
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 3),
        "risk_level": risk,
        "text_hash_sha256": sha256_text(text)
    }

    result.update(checks)

    return result


def make_report(results_df, case_id, examiner, source_name):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    high = int((results_df["risk_level"] == "High").sum())
    medium = int(results_df["risk_level"].isin(["Medium", "Low-Medium"]).sum())
    flagged = results_df[results_df["risk_level"].isin(["High", "Medium", "Low-Medium"])]

    lines = [
        "COS 783 SOCIAL MEDIA FORENSIC ANALYSIS REPORT",
        f"Case ID: {case_id or 'Not specified'}",
        f"Examiner: {examiner or 'Not specified'}",
        f"Evidence source: {source_name}",
        f"Generated: {generated_at}",
        "",
        "SUMMARY",
        f"- Records analysed: {len(results_df)}",
        f"- High risk records: {high}",
        f"- Medium / low-medium risk records: {medium}",
        f"- Flagged records: {len(flagged)}",
        "",
        "METHOD",
        "- Text was cleaned using basic NLP preprocessing.",
        "- A TF-IDF and Logistic Regression model was used for classification.",
        "- VADER sentiment analysis was used to identify negative or positive tone.",
        "- A rule-based forensic layer checked for threat, crime, scam, and toxic indicators.",
        "- SHA-256 hashes were generated for analysed text to support repeatability.",
        "",
        "FLAGGED RECORDS"
    ]

    if flagged.empty:
        lines.append("- No flagged records found.")
    else:
        for index, row in flagged.head(50).iterrows():
            lines.extend([
                f"- Record {index}",
                f"  Risk: {row['risk_level']}",
                f"  AI classification: {row['ai_classification']}",
                f"  Sentiment: {row['sentiment']}",
                f"  Rule detection: {row['rule_detection']}",
                f"  Hash: {row['text_hash_sha256']}",
                f"  Redacted text: {row['redacted_text']}"
            ])

    return "\n".join(lines)


def choose_text_column(columns):
    preferred = ["tweet", "tweet_text", "text", "content", "message", "body", "comment"]
    lower_map = {col.lower(): col for col in columns}

    for item in preferred:
        if item in lower_map:
            return lower_map[item]

    for col in columns:
        lowered = col.lower()
        if any(item in lowered for item in preferred):
            return col

    return columns[0]


def main():
    inject_css()

    model_bundle = load_model_bundle()

    st.markdown(
        """
        <section class="hero">
            <h1>Social Media Threat Analysis Lab</h1>
            <p>
                COS 783 forensic NLP tool for analysing social media posts, identifying harmful content,
                tracking suspicious indicators, scoring risk, and exporting evidence-ready results.
            </p>
            <div class="badge-row">
                <span class="badge">AI classification</span>
                <span class="badge">Sentiment analysis</span>
                <span class="badge">Threat indicators</span>
                <span class="badge">SHA-256 evidence hashing</span>
                <span class="badge">CSV report export</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True
    )

    if model_bundle is None:
        st.error(
            "Model not found. Please run `python train_model.py` first. "
            "This will create `forensic_social_media_model.pkl`."
        )
        st.stop()

    with st.sidebar:
        st.header("Investigation Setup")
        case_id = st.text_input("Case ID", value="COS783-SMA-001")
        examiner = st.text_input("Examiner", value="")
        st.divider()

        st.subheader("Loaded AI Model")
        st.write("Model: TF-IDF + Logistic Regression")
        st.write("Rows used:", model_bundle.get("rows_used", "Unknown"))
        accuracy = model_bundle.get("accuracy", None)
        if accuracy is not None:
            st.write("Test accuracy:", f"{accuracy * 100:.1f}%")

        st.divider()
        st.subheader("Detection Layers")
        st.caption("AI classifier")
        st.caption("VADER sentiment")
        st.caption("Threat/crime/toxicity rules")
        st.caption("Forensic hashing")

    acquisition_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(
        f"""
        <section class="case-strip">
            <div class="case-tile">
                <span>Case Reference</span>
                <strong>{html.escape(case_id or 'Unassigned')}</strong>
            </div>
            <div class="case-tile">
                <span>Examiner</span>
                <strong>{html.escape(examiner or 'Not recorded')}</strong>
            </div>
            <div class="case-tile">
                <span>Model Status</span>
                <strong>Loaded</strong>
            </div>
            <div class="case-tile">
                <span>Analysis Time</span>
                <strong>{acquisition_time}</strong>
            </div>
        </section>
        """,
        unsafe_allow_html=True
    )

    tab_live, tab_csv, tab_review, tab_report = st.tabs(
        ["Live Triage", "Dataset Analysis", "Evidence Review", "Case Report"]
    )

    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "source_name" not in st.session_state:
        st.session_state["source_name"] = "Manual analysis"

    with tab_live:
        st.subheader("Live Social Media Triage")
        st.write("Paste a tweet, comment, or social media post below.")

        example_col, clear_col = st.columns([1, 1])
        with example_col:
            if st.button("Use example post"):
                st.session_state["manual_text"] = "I will attack them tomorrow. They are useless."
        with clear_col:
            if st.button("Clear text"):
                st.session_state["manual_text"] = ""

        manual_text = st.text_area(
            "Post to analyse",
            key="manual_text",
            height=150,
            placeholder="Paste social media text here..."
        )

        if st.button("Analyse Post", type="primary"):
            if not manual_text.strip():
                st.warning("Please enter a post first.")
            else:
                result = analyse_text(manual_text, model_bundle)
                result_df = pd.DataFrame([result])
                st.session_state["results_df"] = result_df
                st.session_state["source_name"] = "Manual input"

                risk = result["risk_level"]
                safe_text = html.escape(result["redacted_text"])

                st.markdown(
                    f"""
                    <div class="tweet-box">
                        <div class="tweet-text">{safe_text}</div>
                        <div>
                            Risk classification:
                            <span class="{risk_class(risk)}">{risk}</span>
                            &nbsp; | &nbsp; AI label:
                            <strong>{html.escape(str(result['ai_classification']))}</strong>
                        </div>
                        <div class="evidence">
                            <strong>AI confidence:</strong> {result['ai_confidence']}%<br>
                            <strong>Sentiment:</strong> {result['sentiment']} ({result['sentiment_score']})<br>
                            <strong>Rule detection:</strong> {result['rule_detection']}<br>
                            <strong>Threat terms:</strong> {result['threat_terms']}<br>
                            <strong>Crime/scam terms:</strong> {result['crime_terms']}<br>
                            <strong>Toxic terms:</strong> {result['toxic_terms']}<br>
                            <strong>SHA-256:</strong> {result['text_hash_sha256']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with tab_csv:
        st.subheader("Dataset Analysis")
        uploaded_file = st.file_uploader("Upload social media CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                st.stop()

            if df.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                st.write("Dataset preview")
                st.dataframe(df.head(10), use_container_width=True)

                columns = list(df.columns)
                default_col = choose_text_column(columns)

                text_column = st.selectbox(
                    "Select text column",
                    columns,
                    index=columns.index(default_col)
                )

                max_rows = st.slider(
                    "Maximum rows to analyse",
                    min_value=50,
                    max_value=min(10000, len(df)),
                    value=min(1000, len(df)),
                    step=50
                )

                if st.button("Run Dataset Analysis", type="primary"):
                    work_df = df.head(max_rows).copy()
                    results = []

                    progress = st.progress(0)

                    for count, text in enumerate(work_df[text_column]):
                        results.append(analyse_text(text, model_bundle))
                        progress.progress((count + 1) / len(work_df))

                    results_df = pd.DataFrame(results)
                    st.session_state["results_df"] = results_df
                    st.session_state["source_name"] = uploaded_file.name

                    st.success("Dataset analysis completed.")

        results_df = st.session_state["results_df"]

        if results_df is not None:
            high_count = int((results_df["risk_level"] == "High").sum())
            medium_count = int(results_df["risk_level"].isin(["Medium", "Low-Medium"]).sum())
            low_count = int((results_df["risk_level"] == "Low").sum())
            flagged_count = int(results_df["risk_level"].isin(["High", "Medium", "Low-Medium"]).sum())

            metric_cols = st.columns(4)
            with metric_cols[0]:
                render_metric("Records Analysed", len(results_df), st.session_state["source_name"])
            with metric_cols[1]:
                render_metric("Flagged Records", flagged_count, "High, medium and low-medium")
            with metric_cols[2]:
                render_metric("High Risk", high_count, "Priority review")
            with metric_cols[3]:
                render_metric("Flag Rate", f"{flagged_count / max(1, len(results_df)) * 100:.1f}%", "Forensic triage")

            chart_data = results_df["risk_level"].value_counts().reset_index()
            chart_data.columns = ["Risk Level", "Records"]

            left, right = st.columns([1, 1])
            with left:
                st.subheader("Risk Distribution")
                st.bar_chart(chart_data, x="Risk Level", y="Records")
            with right:
                st.subheader("Top AI Classifications")
                label_data = results_df["ai_classification"].value_counts().reset_index()
                label_data.columns = ["Classification", "Records"]
                st.dataframe(label_data, use_container_width=True, hide_index=True)

            st.subheader("Analysed Evidence Table")
            table_cols = [
                "redacted_text",
                "risk_level",
                "ai_classification",
                "ai_confidence",
                "sentiment",
                "rule_detection",
                "text_hash_sha256"
            ]
            st.dataframe(
                results_df[table_cols].sort_values("risk_level"),
                use_container_width=True,
                hide_index=True
            )

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)

            st.download_button(
                "Download analysed CSV",
                csv_buffer.getvalue(),
                file_name="cos783_analysed_social_media.csv",
                mime="text/csv"
            )

    with tab_review:
        st.subheader("Evidence Review")

        results_df = st.session_state["results_df"]

        if results_df is None:
            st.info("Run a live analysis or upload a CSV first.")
        else:
            risk_filter = st.multiselect(
                "Risk filter",
                ["High", "Medium", "Low-Medium", "Low"],
                default=["High", "Medium", "Low-Medium"]
            )

            search_term = st.text_input("Search redacted text")

            review_df = results_df[results_df["risk_level"].isin(risk_filter)].copy()

            if search_term:
                review_df = review_df[
                    review_df["redacted_text"].str.contains(
                        search_term,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

            if review_df.empty:
                st.info("No records match the current filters.")
            else:
                for index, row in review_df.head(30).iterrows():
                    safe_text = html.escape(str(row["redacted_text"]))
                    risk = str(row["risk_level"])

                    st.markdown(
                        f"""
                        <div class="tweet-box">
                            <div class="tweet-text">{safe_text}</div>
                            <div>
                                Risk:
                                <span class="{risk_class(risk)}">{html.escape(risk)}</span>
                                &nbsp; | &nbsp; AI:
                                <strong>{html.escape(str(row['ai_classification']))}</strong>
                                &nbsp; | &nbsp; Sentiment:
                                <strong>{html.escape(str(row['sentiment']))}</strong>
                            </div>
                            <div class="evidence">
                                <strong>Record:</strong> {index}<br>
                                <strong>Rule detection:</strong> {html.escape(str(row['rule_detection']))}<br>
                                <strong>Threat terms:</strong> {html.escape(str(row['threat_terms']))}<br>
                                <strong>Crime/scam terms:</strong> {html.escape(str(row['crime_terms']))}<br>
                                <strong>Toxic terms:</strong> {html.escape(str(row['toxic_terms']))}<br>
                                <strong>Target terms:</strong> {html.escape(str(row['target_terms']))}<br>
                                <strong>SHA-256:</strong> {html.escape(str(row['text_hash_sha256']))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with tab_report:
        st.subheader("Case Report Export")

        results_df = st.session_state["results_df"]

        if results_df is None:
            st.info("Analyse text or a dataset first to generate a case report.")
        else:
            report = make_report(
                results_df,
                case_id,
                examiner,
                st.session_state["source_name"]
            )

            st.text_area("Generated report", report, height=430)

            st.download_button(
                "Download case report",
                report,
                file_name=f"{case_id or 'case'}_social_media_report.txt",
                mime="text/plain"
            )

            left, right = st.columns([1, 1])
            with left:
                st.markdown(
                    """
                    <div class="custody-panel">
                        <h3>Evidence Handling Notes</h3>
                        <p>Preserve the original CSV separately before analysis.</p>
                        <p>Use SHA-256 hashes to support repeatability of reviewed artefacts.</p>
                        <p>AI results are triage outputs and should be reviewed by a human.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with right:
                st.markdown(
                    """
                    <div class="custody-panel">
                        <h3>Method Summary</h3>
                        <p>The tool combines supervised machine learning, sentiment analysis, and rule-based indicators.</p>
                        <p>This makes the system more explainable for a digital forensics assignment.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
