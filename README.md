# Digital Evidence Hate Speech Lab

A Streamlit application for examining Twitter/X-style CSV artefacts as part of a digital forensics hate-speech detection workflow.

## Features

- Upload a CSV dataset or use the included sample data.
- Train a supervised AI model from labelled tweet data.
- Type a sentence manually and classify it as low, medium, or high risk.
- Select the tweet text column and optional existing label column.
- Score tweets using a transparent baseline detector.
- Review high and medium risk records.
- Generate SHA-256 hashes for analysed tweet text.
- Redact handles and URLs in evidence previews.
- Track case metadata such as case ID, examiner, evidence source, and acquisition time.
- Export an analysed evidence CSV and a plain-text forensic case report.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Dataset Format

The app works with any CSV that has a text column and, for AI training, a label column. It will automatically look for common text column names such as:

- `tweet`
- `text`
- `content`
- `message`
- `body`

Optional columns such as `label`, `class`, `category`, or `hate_speech` can be selected for comparison.

For training, choose the label column and then select which label means hate speech. Example labels:

- `hate`
- `non-hate`
- `offensive`
- `normal`

## Note

The AI model uses TF-IDF text features with Logistic Regression. This is suitable for an explainable class project and forensic triage demonstration. For production-quality detection, use a larger validated dataset, test for bias, and document false positives and false negatives.
