# Agentic AI Data Scientist

An agentic system that automates the major stages of a data science workflow
from a natural-language instruction and an uploaded dataset — combining
LLM reasoning with deterministic Python/ML tools rather than relying on
generated text for anything numeric. Once trained, the model is saved and
can predict on brand-new records through a web UI or CLI.

## Architecture

```
main.py            CLI entry point — runs the pipeline, saves the report + trained model
predict.py          Standalone CLI — load a saved model and predict on new records
app.py               Streamlit web UI — upload, run, browse results, live predictions
orchestrator.py     Wires agents together based on the instruction
agents.py           7 specialized agents (reasoning + delegation)
tools.py            Deterministic pandas / sklearn / matplotlib / sqlite functions
llm.py              Reasoner: live Claude API call, or offline heuristic fallback
report.py           Renders everything into one animated, styled HTML report
```

### Why this split matters
- **Agents reason, tools compute.** An agent decides *what* to do (which
  column is the target, which task type, which models to try); `tools.py`
  actually does it with pandas/sklearn. This means every number in the
  final report is reproducible and auditable — never hallucinated.
- **`Reasoner` is swappable.** With `ANTHROPIC_API_KEY` set, every judgment
  call agents make can route through a real Claude API call
  (`api.anthropic.com/v1/messages`). Without it, deterministic heuristics
  keep the whole pipeline runnable offline (useful for demos/CI/grading).
- **Every agent keeps a trace.** `agent.trace` is a list of natural-language
  log lines describing every decision and action, which becomes section 6
  of the report — a transparent audit trail, not a black box.
- **The trained model isn't thrown away.** After training, the best pipeline
  (preprocessing + model) is saved to `outputs/trained_model.joblib`, so it
  can be reused to predict on new records without retraining.

## The 7 Agents

| Agent | Responsibility |
|---|---|
| `DataUnderstandingAgent` | Profiles the dataset (shape, dtypes, missingness, cardinality); infers the target column from the instruction or naming conventions |
| `CleaningAgent` | Drops duplicates, ID-like columns, high-cardinality free text, and constant/near-empty columns; imputes missing values (median/mode); caps outliers via 3×IQR |
| `FeatureEngineeringAgent` | Detects datetime-like text columns and decomposes them (year/month/day-of-week) |
| `EDAAgent` | Correlation heatmap, distribution plots, missingness plot, target-relationship plots, ranked correlation insights |
| `SQLAgent` | Runs SQL directly against the dataframe via an in-memory SQLite table, extracted from the instruction or passed explicitly |
| `ModelSelectionTrainingAgent` | Infers classification vs. regression, trains multiple candidate models (LogReg/RF/GBM/SVM or their regressors), evaluates on a held-out split, and picks the best by primary metric |
| `ReportAgent` | Compiles every stage's output — tables, plots, agent traces — into one animated HTML report |

## Running it

### Easiest: one-click scripts
- **Windows**: double-click `run.bat`. First run sets up a virtual
  environment and installs dependencies; every run after that starts
  the app directly since it detects the existing environment.
- **Mac/Linux**: `bash run.sh`

Both launch the Streamlit web UI at `http://localhost:8501`.

### Manual setup
```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

### Web UI (Streamlit)
Upload a dataset (or use the bundled sample), type an instruction and
optional SQL query, click **Run agent pipeline**, and browse results
across sections: Data Profile, Cleaning, EDA, SQL, Modeling, **Predict New
Data**, Agent Trace, and a downloadable full HTML report. The Predict New
Data section auto-generates an input form from the trained model's
features and returns a live prediction with a probability percentage.
There's also an optional field in the sidebar to paste an Anthropic API
key for live LLM reasoning.

### CLI
```bash
python main.py \
  --data your_dataset.csv \
  --instruction "Predict customer churn" \
  --sql "SELECT plan_type, COUNT(*) FROM data GROUP BY plan_type;" \
  --out outputs/report.html
```
This also saves the trained model to `outputs/trained_model.joblib`.

### Predicting on new data (after training)
```bash
# single record
python predict.py --model outputs/trained_model.joblib \
  --input '{"age": 45, "monthly_income": 52000, "tenure_years": 0.5, "plan_type": "Basic", "support_calls": 6, "signup_date_year": 2024, "signup_date_month": 3, "signup_date_dayofweek": 2}'

# batch, from a CSV
python predict.py --model outputs/trained_model.joblib --csv new_customers.csv --out predictions.csv
```
The exact fields required depend on your dataset — run with no `--input`/`--csv`
flag first and it will print the feature list the loaded model expects.

### Enabling live LLM reasoning
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --data your_dataset.csv --instruction "..."
```
Without a key, agents fall back to deterministic heuristics — the pipeline
still runs fully end-to-end offline.

## Demo included

`sample_customer_churn.csv` — a synthetic 600-row customer dataset (age,
income, tenure, plan type, support calls, signup date, churn label) with
injected missing values and a duplicate row, so cleaning/EDA/modeling all
have real work to do.

## Extending it

- **New tool stage** (e.g. time-series forecasting, clustering): add a
  function to `tools.py`, wrap it in a new `Agent` subclass in `agents.py`,
  add it to `Orchestrator.run()`.
- **Smarter reasoning**: replace the heuristic branches in
  `DataUnderstandingAgent._infer_target` or model selection with a call to
  `self.reasoner.ask(...)`, once `ANTHROPIC_API_KEY` is set — the plumbing
  is already there.
- **Different report format**: swap `report.py` for a PDF/PPTX generator;
  the `context` dict passed to `ReportAgent` already has everything needed.

## Developed by
Sehrab Showket Shah
