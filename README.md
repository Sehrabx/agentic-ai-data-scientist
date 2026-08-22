# Agentic AI Data Scientist

An agentic system that automates the major stages of a data science workflow
from a natural-language instruction and an uploaded dataset — combining
LLM reasoning with deterministic Python/ML tools rather than relying on
generated text for anything numeric.

## Architecture

```
main.py            CLI entry point
orchestrator.py     Wires agents together based on the instruction
agents.py           7 specialized agents (reasoning + delegation)
tools.py            Deterministic pandas / sklearn / matplotlib / sqlite functions
llm.py              Reasoner: live Claude API call, or offline heuristic fallback
report.py           Renders everything into one styled HTML report
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

## The 7 Agents

| Agent | Responsibility |
|---|---|
| `DataUnderstandingAgent` | Profiles the dataset (shape, dtypes, missingness, cardinality); infers the target column from the instruction or naming conventions |
| `CleaningAgent` | Drops duplicates/constant/near-empty columns, imputes missing values (median/mode), caps outliers via 3×IQR |
| `FeatureEngineeringAgent` | Detects datetime-like text columns and decomposes them (year/month/day-of-week) |
| `EDAAgent` | Correlation heatmap, distribution plots, missingness plot, target-relationship plots, ranked correlation insights |
| `SQLAgent` | Runs SQL directly against the dataframe via an in-memory SQLite table, extracted from the instruction or passed explicitly |
| `ModelSelectionTrainingAgent` | Infers classification vs. regression, trains multiple candidate models (LogReg/RF/GBM/SVM or their regressors), evaluates on a held-out split, and picks the best by primary metric |
| `ReportAgent` | Compiles every stage's output — tables, plots, agent traces — into one HTML report |

## Running it

### CLI
```bash
pip install -r requirements.txt

python main.py \
  --data your_dataset.csv \
  --instruction "Predict customer churn" \
  --sql "SELECT plan_type, COUNT(*) FROM data GROUP BY plan_type;" \
  --out outputs/report.html
```

### Web UI (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens a browser app where you upload a dataset (or use the bundled sample),
type an instruction and optional SQL query, click **Run agent pipeline**,
and browse results across tabs: Data Profile, Cleaning, EDA, SQL, Modeling,
Agent Trace, and a downloadable full HTML report. There's also an optional
field in the sidebar to paste an Anthropic API key for live LLM reasoning.

To enable live LLM-driven reasoning at each decision point instead of the
offline heuristics:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --data your_dataset.csv --instruction "..."
```

## Demo included

`sample_customer_churn.csv` — a synthetic 600-row customer dataset (age,
income, tenure, plan type, support calls, signup date, churn label) with
injected missing values and a duplicate row, so cleaning/EDA/modeling all
have real work to do. Run the command above on it to reproduce
`outputs/report.html`.

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

## Known limitations (worth stating in a viva/report)
- Target-column inference is currently heuristic when no LLM key is set —
  works well on common naming conventions but can misfire on unusual schemas.
- Model search is a fixed candidate set (no hyperparameter tuning) — swap in
  `GridSearchCV`/`Optuna` for a production version.
- SQL agent extracts queries via regex from the instruction; for robust NL→SQL
  you'd route through the live `Reasoner` to generate the query itself.
