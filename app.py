import hashlib
import html
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ImportError:
    TfidfVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    classification_report = None
    train_test_split = None
    Pipeline = None


st.set_page_config(
    page_title="Tweet Hate Speech Forensics",
    page_icon="DF",
    layout="wide",
    initial_sidebar_state="expanded",
)


HATE_TERMS = {
    "hate",
    "kill",
    "terrorist",
    "vermin",
    "animal",
    "filth",
    "invader",
    "parasite",
    "exterminate",
    "deport",
    "inferior",
    "scum",
    "traitor",
    "trash",
}

THREAT_TERMS = {
    "attack",
    "destroy",
    "shoot",
    "stab",
    "bomb",
    "burn",
    "hurt",
    "violence",
    "threat",
}

TARGET_TERMS = {
    "race",
    "religion",
    "gender",
    "immigrant",
    "foreigner",
    "minority",
    "community",
    "group",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --case-navy: #18202b;
                --case-ink: #253040;
                --case-border: #d8dee8;
                --case-panel: #f7f9fc;
                --case-accent: #2d6cdf;
                --case-danger: #b42318;
                --case-warn: #b54708;
                --case-ok: #067647;
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
                background: linear-gradient(135deg, #172033 0%, #263449 56%, #36546e 100%);
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 8px;
                padding: 28px 30px;
                color: #ffffff;
                box-shadow: 0 18px 45px rgba(19, 33, 55, 0.18);
                margin-bottom: 22px;
            }

            .hero h1 {
                margin: 0 0 8px 0;
                font-size: 2.3rem;
                line-height: 1.1;
                color: #ffffff;
            }

            .hero p {
                margin: 0;
                color: #dce7f6;
                max-width: 960px;
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
                border-radius: 999px;
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

            .tweet-box {
                border: 1px solid var(--case-border);
                border-left: 5px solid var(--case-accent);
                background: #ffffff;
                border-radius: 8px;
                padding: 18px;
                margin-bottom: 14px;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def redact_handles(text: str) -> str:
    text = re.sub(r"@\w+", "@[redacted]", text)
    return re.sub(r"https?://\S+", "[url]", text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_tweet(text: str, threshold: int) -> dict[str, object]:
    tokens = tokenize(text)
    token_set = set(tokens)
    hate_hits = sorted(token_set.intersection(HATE_TERMS))
    threat_hits = sorted(token_set.intersection(THREAT_TERMS))
    target_hits = sorted(token_set.intersection(TARGET_TERMS))
    uppercase_ratio = sum(1 for char in text if char.isupper()) / max(1, sum(1 for char in text if char.isalpha()))
    exclamation_count = text.count("!")

    raw_score = (
        len(hate_hits) * 3
        + len(threat_hits) * 2
        + len(target_hits)
        + min(2, exclamation_count)
        + (1 if uppercase_ratio > 0.35 and len(text) > 12 else 0)
    )

    if raw_score >= threshold + 3:
        risk = "High"
    elif raw_score >= threshold:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "risk_score": raw_score,
        "risk_level": risk,
        "hate_terms": ", ".join(hate_hits) if hate_hits else "None",
        "threat_terms": ", ".join(threat_hits) if threat_hits else "None",
        "target_terms": ", ".join(target_hits) if target_hits else "None",
        "text_hash_sha256": sha256_text(text),
        "redacted_text": redact_handles(text),
    }


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv("sample_tweets.csv")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def choose_default_column(columns: list[str], candidates: list[str]) -> str:
    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    for column in columns:
        lowered = column.lower()
        if any(candidate in lowered for candidate in candidates):
            return column
    return columns[0]


def sklearn_available() -> bool:
    return all(
        item is not None
        for item in [
            TfidfVectorizer,
            LogisticRegression,
            accuracy_score,
            classification_report,
            train_test_split,
            Pipeline,
        ]
    )


def prepare_training_data(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    hate_label: str,
) -> tuple[pd.Series, pd.Series]:
    training = df[[text_column, label_column]].copy()
    training[text_column] = training[text_column].apply(normalize_text)
    training[label_column] = training[label_column].apply(normalize_text)
    training = training[(training[text_column] != "") & (training[label_column] != "")]
    labels = training[label_column].astype(str)
    y = (labels == str(hate_label)).astype(int)
    return training[text_column], y


def train_hate_speech_model(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    hate_label: str,
) -> dict[str, object]:
    if not sklearn_available():
        raise RuntimeError("scikit-learn is not installed. Run: pip install -r requirements.txt")

    texts, labels = prepare_training_data(df, text_column, label_column, hate_label)
    if len(texts) < 6:
        raise ValueError("The model needs at least 6 labelled rows to train.")
    if labels.nunique() < 2:
        raise ValueError("The selected label must contain both hate-speech and non-hate examples.")

    stratify = labels if labels.value_counts().min() >= 2 else None
    test_size = 0.3 if len(texts) >= 10 else 0.4
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["not hate speech", "hate speech"],
            zero_division=0,
        ),
        "training_rows": int(len(texts)),
        "test_rows": int(len(x_test)),
        "hate_rows": int(labels.sum()),
        "non_hate_rows": int((labels == 0).sum()),
        "hate_label": hate_label,
    }


def ai_risk_from_probability(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"


def predict_with_ai_model(text: str, model_bundle: dict[str, object]) -> dict[str, object]:
    model = model_bundle["model"]
    probability = float(model.predict_proba([text])[0][1])
    baseline = score_tweet(text, threshold=4)
    risk = ai_risk_from_probability(probability)

    return {
        **baseline,
        "risk_score": round(probability * 100, 1),
        "risk_level": risk,
        "ai_probability": probability,
        "model_method": "Trained TF-IDF + Logistic Regression",
    }


def classify_dataset(
    df: pd.DataFrame,
    text_column: str,
    threshold: int,
    model_bundle: dict[str, object] | None = None,
) -> pd.DataFrame:
    working = df.copy()
    working[text_column] = working[text_column].apply(normalize_text)
    if model_bundle is None:
        scores = working[text_column].apply(lambda text: score_tweet(text, threshold)).apply(pd.Series)
        scores["model_method"] = "Keyword baseline"
    else:
        scores = working[text_column].apply(lambda text: predict_with_ai_model(text, model_bundle)).apply(pd.Series)
    return pd.concat([working, scores], axis=1)


def make_report(
    df: pd.DataFrame,
    text_column: str,
    case_id: str,
    investigator: str,
    model_status: str,
) -> str:
    flagged = df[df["risk_level"].isin(["High", "Medium"])]
    high = int((df["risk_level"] == "High").sum())
    medium = int((df["risk_level"] == "Medium").sum())
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "DIGITAL FORENSICS HATE SPEECH SCREENING REPORT",
        f"Case ID: {case_id or 'Not specified'}",
        f"Investigator: {investigator or 'Not specified'}",
        f"Generated: {generated_at}",
        "",
        "Scope",
        f"- Source records analysed: {len(df)}",
        f"- Tweet/text column: {text_column}",
        f"- Flagged records: {len(flagged)}",
        f"- High risk records: {high}",
        f"- Medium risk records: {medium}",
        "",
        "Method",
        f"- Detection method used: {model_status}.",
        "- If a labelled dataset is selected, the app trains a TF-IDF and Logistic Regression classifier.",
        "- Keyword indicators are retained as supporting triage features and explainability notes.",
        "- SHA-256 hashes are generated from the analysed tweet text to support repeatability.",
        "- User handles and URLs are redacted in preview text to reduce accidental exposure.",
        "",
        "Flagged Evidence",
    ]

    if flagged.empty:
        lines.append("- No medium or high risk records were flagged.")
    else:
        for index, row in flagged.head(50).iterrows():
            lines.extend(
                [
                    f"- Record {index}",
                    f"  Risk: {row['risk_level']} ({row['risk_score']})",
                    f"  Hash: {row['text_hash_sha256']}",
                    f"  Hate terms: {row['hate_terms']}",
                    f"  Threat terms: {row['threat_terms']}",
                    f"  Target terms: {row['target_terms']}",
                    f"  Redacted text: {row['redacted_text']}",
                ]
            )

    return "\n".join(lines)


def render_metric(label: str, value: object, helper: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{helper}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_class(level: str) -> str:
    return {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(level, "risk-low")


def risk_explanation(result: dict[str, object]) -> str:
    level = str(result["risk_level"])
    if "ai_probability" in result:
        probability = float(result["ai_probability"]) * 100
        return f"The trained AI model estimates a {probability:.1f}% hate-speech probability for this sentence."
    if level == "High":
        return "This sentence should be prioritised for human review because it contains several strong indicators."
    if level == "Medium":
        return "This sentence contains concerning indicators and should be reviewed in context."
    return "This sentence has few or no baseline indicators. It is low risk under the current threshold."


def render_manual_test(threshold: int, model_bundle: dict[str, object] | None) -> None:
    st.subheader("Manual Hate Speech Test")
    if model_bundle is None:
        st.write("Type a sentence or tweet below to test it with the keyword baseline detector.")
        st.warning("Train the AI model from a labelled dataset to enable machine-learning predictions.")
    else:
        st.write("Type a sentence or tweet below to test it with the trained AI hate-speech classifier.")

    example_col, clear_col = st.columns([1, 1])
    with example_col:
        if st.button("Use high-risk example"):
            st.session_state["manual_sentence"] = "They are parasites and we should attack their community!"
    with clear_col:
        if st.button("Clear sentence"):
            st.session_state["manual_sentence"] = ""

    sentence = st.text_area(
        "Sentence to analyse",
        key="manual_sentence",
        height=140,
        placeholder="Type a tweet or sentence here...",
    )

    if not sentence.strip():
        st.info("Enter a sentence to see the risk level, score, indicators, and evidence hash.")
        return

    if model_bundle is None:
        result = score_tweet(sentence, threshold)
        model_name = "Keyword baseline"
        score_label = "Risk score"
    else:
        result = predict_with_ai_model(sentence, model_bundle)
        model_name = str(result["model_method"])
        score_label = "AI probability"

    risk = str(result["risk_level"])
    score = result["risk_score"]
    safe_text = html.escape(str(result["redacted_text"]))

    st.markdown(
        f"""
        <div class="tweet-box">
            <div class="tweet-text">{safe_text}</div>
            <div>
                Classification:
                <span class="{risk_class(risk)}">{risk} risk</span>
                &nbsp; | &nbsp; {score_label}: <strong>{score}</strong>
            </div>
            <div class="evidence">
                <strong>Model:</strong> {model_name}<br>
                <strong>Interpretation:</strong> {risk_explanation(result)}<br>
                <strong>SHA-256:</strong> {result['text_hash_sha256']}<br>
                <strong>Hate indicators:</strong> {result['hate_terms']}<br>
                <strong>Threat indicators:</strong> {result['threat_terms']}<br>
                <strong>Target indicators:</strong> {result['target_terms']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.metric("Risk level", risk)
    with detail_cols[1]:
        st.metric(score_label, score)
    with detail_cols[2]:
        st.metric("Model", model_name)

    st.caption(
        "This is a triage tool for assignment demonstration. "
        "Final decisions should include human review and context."
    )


def main() -> None:
    inject_css()

    st.markdown(
        """
        <section class="hero">
            <h1>Tweet Hate Speech Digital Forensics Dashboard</h1>
            <p>
                Analyse Twitter/X-style datasets, identify potentially hateful or threatening language,
                preserve repeatable evidence hashes, and prepare concise findings for investigation notes.
            </p>
            <div class="badge-row">
                <span class="badge">Dataset triage</span>
                <span class="badge">AI model training</span>
                <span class="badge">Evidence hashing</span>
                <span class="badge">Report export</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Case Setup")
        case_id = st.text_input("Case ID", value="COS783-HS-001")
        investigator = st.text_input("Investigator", value="")
        threshold = st.slider("Flagging threshold", min_value=2, max_value=10, value=4)
        uploaded_file = st.file_uploader("Upload tweet dataset", type=["csv"])
        st.caption("CSV files work best when they include a tweet, text, content, or message column.")

        st.divider()
        st.subheader("Baseline Dictionary")
        st.write(f"Hate indicators: {len(HATE_TERMS)}")
        st.write(f"Threat indicators: {len(THREAT_TERMS)}")
        st.write(f"Target indicators: {len(TARGET_TERMS)}")

    try:
        if uploaded_file is not None:
            df = read_uploaded_csv(uploaded_file)
            source_name = uploaded_file.name
        else:
            df = load_sample_data()
            source_name = "sample_tweets.csv"
    except Exception as exc:
        st.error(f"Could not load the CSV file: {exc}")
        st.stop()

    if df.empty:
        st.warning("The dataset is empty. Upload a CSV with tweet text to continue.")
        st.stop()

    columns = list(df.columns)
    default_text = choose_default_column(columns, ["tweet", "text", "content", "message", "body"])

    top_left, top_right = st.columns([2, 1])
    with top_left:
        text_column = st.selectbox("Tweet text column", columns, index=columns.index(default_text))
    with top_right:
        label_candidates = ["None"] + columns
        default_label = "None"
        for candidate in columns:
            if candidate.lower() in {"label", "class", "category", "hate_speech", "sentiment"}:
                default_label = candidate
                break
        label_column = st.selectbox(
            "Existing label column",
            label_candidates,
            index=label_candidates.index(default_label),
        )

    model_bundle = None
    model_status = "Keyword baseline"
    if label_column != "None":
        label_values = sorted(
            value
            for value in df[label_column].dropna().astype(str).str.strip().unique().tolist()
            if value
        )
        guessed_hate_label = 0
        for position, value in enumerate(label_values):
            lowered = value.lower()
            if "hate" in lowered and "non" not in lowered and "not" not in lowered:
                guessed_hate_label = position
                break

        with st.expander("AI Model Training", expanded=True):
            st.write("Train a supervised NLP model from the selected dataset labels.")
            if not sklearn_available():
                st.error("scikit-learn is required for AI training. Run `pip install -r requirements.txt`.")
            elif len(label_values) < 2:
                st.warning("The selected label column needs at least two different labels.")
            else:
                hate_label = st.selectbox(
                    "Which label means hate speech?",
                    label_values,
                    index=guessed_hate_label,
                )
                train_model = st.checkbox("Train AI model from this dataset", value=True)
                if train_model:
                    try:
                        model_bundle = train_hate_speech_model(df, text_column, label_column, hate_label)
                        model_status = "Trained AI model"
                        metric_a, metric_b, metric_c, metric_d = st.columns(4)
                        with metric_a:
                            st.metric("Training rows", model_bundle["training_rows"])
                        with metric_b:
                            st.metric("Hate examples", model_bundle["hate_rows"])
                        with metric_c:
                            st.metric("Non-hate examples", model_bundle["non_hate_rows"])
                        with metric_d:
                            st.metric("Test accuracy", f"{model_bundle['accuracy'] * 100:.1f}%")
                        with st.expander("Classification report"):
                            st.code(str(model_bundle["report"]), language="text")
                    except Exception as exc:
                        st.error(f"AI model training could not complete: {exc}")
                        model_bundle = None
    else:
        st.info("Select a label column to train the AI model. Until then, the app uses the keyword baseline.")

    results = classify_dataset(df, text_column, threshold, model_bundle)
    flagged = results[results["risk_level"].isin(["High", "Medium"])]
    high_count = int((results["risk_level"] == "High").sum())
    medium_count = int((results["risk_level"] == "Medium").sum())
    low_count = int((results["risk_level"] == "Low").sum())

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric("Records Analysed", len(results), f"Source: {source_name}")
    with metric_cols[1]:
        render_metric("Flagged Records", len(flagged), "Medium and high risk")
    with metric_cols[2]:
        render_metric("High Risk", high_count, "Priority evidence review")
    with metric_cols[3]:
        rate = f"{(len(flagged) / max(1, len(results)) * 100):.1f}%"
        render_metric("Flag Rate", rate, model_status)

    tab_manual, tab_overview, tab_evidence, tab_dataset, tab_report = st.tabs(
        ["Manual Test", "Overview", "Evidence Review", "Dataset Explorer", "Report"]
    )

    with tab_manual:
        render_manual_test(threshold, model_bundle)

    with tab_overview:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Risk Distribution")
            chart_data = pd.DataFrame(
                {
                    "Risk Level": ["High", "Medium", "Low"],
                    "Records": [high_count, medium_count, low_count],
                }
            )
            st.bar_chart(chart_data, x="Risk Level", y="Records", color="#2d6cdf")

        with right:
            st.subheader("Most Common Trigger Terms")
            term_counts: dict[str, int] = {}
            for column in ["hate_terms", "threat_terms", "target_terms"]:
                for value in results[column]:
                    for term in str(value).split(", "):
                        if term and term != "None":
                            term_counts[term] = term_counts.get(term, 0) + 1

            if term_counts:
                terms_df = (
                    pd.DataFrame(
                        [{"Term": term, "Count": count} for term, count in term_counts.items()]
                    )
                    .sort_values("Count", ascending=False)
                    .head(12)
                )
                st.dataframe(terms_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trigger terms were found at the current threshold.")

        st.subheader("Forensic Notes")
        st.markdown(
            """
            This dashboard is designed for supervised AI triage. It can train from labelled tweets,
            highlight records that may need closer human review, store a repeatable text hash for each
            analysed record, and redact handles and URLs in the evidence preview. For formal work, keep
            the original dataset unchanged and document any preprocessing steps.
            """
        )

    with tab_evidence:
        st.subheader("Flagged Tweet Review")
        risk_filter = st.multiselect(
            "Risk level filter",
            ["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
        search_term = st.text_input("Search within tweet text")

        review_df = results[results["risk_level"].isin(risk_filter)]
        if search_term:
            review_df = review_df[
                review_df[text_column].str.contains(search_term, case=False, na=False, regex=False)
            ]
        review_df = review_df.sort_values(["risk_score"], ascending=False)

        if review_df.empty:
            st.info("No records match the current review filters.")
        else:
            for index, row in review_df.head(25).iterrows():
                safe_text = html.escape(str(row["redacted_text"]))
                st.markdown(
                    f"""
                    <div class="tweet-box">
                        <div class="tweet-text">{safe_text}</div>
                        <div>
                            Risk:
                            <span class="{risk_class(row['risk_level'])}">
                                {row['risk_level']} ({row['risk_score']})
                            </span>
                        </div>
                        <div class="evidence">
                            <strong>Record:</strong> {index}<br>
                            <strong>SHA-256:</strong> {row['text_hash_sha256']}<br>
                            <strong>Hate indicators:</strong> {row['hate_terms']}<br>
                            <strong>Threat indicators:</strong> {row['threat_terms']}<br>
                            <strong>Target indicators:</strong> {row['target_terms']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_dataset:
        st.subheader("Dataset Explorer")
        display_columns = [
            text_column,
            "risk_level",
            "risk_score",
            "hate_terms",
            "threat_terms",
            "target_terms",
            "text_hash_sha256",
        ]
        if "ai_probability" in results.columns:
            display_columns.insert(3, "ai_probability")
        if label_column != "None" and label_column not in display_columns:
            display_columns.insert(1, label_column)
        st.dataframe(
            results[display_columns].sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=False,
        )

        csv_buffer = io.StringIO()
        results.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download analysed CSV",
            csv_buffer.getvalue(),
            file_name="analysed_tweet_hate_speech.csv",
            mime="text/csv",
        )

    with tab_report:
        st.subheader("Forensic Report Export")
        report = make_report(results, text_column, case_id, investigator, model_status)
        st.text_area("Generated report", report, height=430)
        st.download_button(
            "Download report",
            report,
            file_name=f"{case_id or 'case'}_hate_speech_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
