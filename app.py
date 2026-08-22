"""
app.py — Streamlit UI for the Agentic AI Data Scientist.

Run with:
    streamlit run app.py

Lets a user upload a dataset, type a natural-language instruction (and
optionally a SQL query), watch each agent's live trace as it runs, and
download the final HTML report.
"""
import os
import tempfile
import streamlit as st
import pandas as pd

from orchestrator import Orchestrator
import tools

st.set_page_config(page_title="Agentic AI Data Scientist", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- STYLE
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ---- animated gradient background behind the whole app ---- */
  .stApp {
      background: linear-gradient(135deg, #0b0f1a, #101528 55%, #0b0f1a);
      background-size: 220% 220%;
      animation: meshShift 20s ease-in-out infinite;
  }
  @keyframes meshShift {
      0%   { background-position: 0% 0%; }
      50%  { background-position: 100% 100%; }
      100% { background-position: 0% 0%; }
  }
  .stApp::before {
      content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background:
        radial-gradient(circle at 12% 15%, rgba(176,58,46,0.28), transparent 42%),
        radial-gradient(circle at 88% 12%, rgba(26,75,140,0.28), transparent 42%),
        radial-gradient(circle at 50% 90%, rgba(176,58,46,0.18), transparent 48%);
  }

  /* ---- hero header ---- */
  .hero { text-align: center; padding: 18px 0 8px; animation: fadeInDown 0.8s ease; }
  .hero h1 {
      font-family: 'Playfair Display', Georgia, serif; font-size: 2.6rem; font-weight: 900;
      background: linear-gradient(90deg, #ff6a5c, #ff9a8b 35%, #6ea8ff 70%, #3a6fd8);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
      margin-bottom: 4px; letter-spacing: -0.5px;
  }
  .hero p { color: #b8c0d4; font-size: 1rem; max-width: 700px; margin: 0 auto; }
  @keyframes fadeInDown { from { opacity: 0; transform: translateY(-16px); } to { opacity: 1; transform: translateY(0); } }

  /* ---- glass content cards ---- */
  .glass {
      background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.09);
      border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; backdrop-filter: blur(10px);
      box-shadow: 0 20px 50px rgba(0,0,0,0.35); animation: fadeUp 0.6s ease;
  }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }

  h1, h2, h3 { color: #f1f2f6 !important; font-family: 'Playfair Display', Georgia, serif !important; }
  p, label, .stMarkdown, .stCaption { color: #cfd4e2; }

  /* ---- metric cards ---- */
  .metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 22px; }
  .metric-card {
      flex: 1; min-width: 160px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
      border-radius: 14px; padding: 16px 18px; text-align: center; backdrop-filter: blur(8px);
      transition: transform 0.25s ease, box-shadow 0.25s ease; animation: fadeUp 0.6s ease;
  }
  .metric-card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(176,58,46,0.25); }
  .metric-card .label { font-size: 12px; color: #9aa3ba; text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 4px;
      background: linear-gradient(90deg, #ff8a75, #6ea8ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

  /* ---- agent trace cards ---- */
  .agent-card {
      background: rgba(255,255,255,0.05); border-left: 3px solid #b03a2e; border-radius: 8px;
      padding: 12px 18px; margin-bottom: 10px; transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .agent-card:hover { transform: translateX(5px); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }
  .agent-card h4 { margin: 0 0 6px 0; color: #ff9a8b !important; font-family: 'Inter', sans-serif !important; font-size: 15px; }
  .agent-card div { color: #d6dbe8; font-size: 13.5px; padding: 2px 0; }

  /* ---- buttons ---- */
  .stButton>button, .stDownloadButton>button {
      background: linear-gradient(90deg, #b03a2e, #8e2f24); color: white; border-radius: 10px; border: none;
      padding: 0.55em 1.6em; font-weight: 600; transition: transform 0.2s ease, box-shadow 0.2s ease;
      box-shadow: 0 6px 18px rgba(176,58,46,0.35);
  }
  .stButton>button:hover, .stDownloadButton>button:hover {
      transform: translateY(-2px) scale(1.02); box-shadow: 0 10px 24px rgba(176,58,46,0.5); color: white;
  }

  /* ---- pill-style section nav (radio) ---- */
  div[data-testid="stRadio"] > div { gap: 8px; flex-wrap: wrap; }
  div[data-testid="stRadio"] label {
      background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 999px; padding: 6px 16px !important; transition: all 0.25s ease; cursor: pointer;
  }
  div[data-testid="stRadio"] label:hover { background: rgba(176,58,46,0.25); border-color: #b03a2e; }
  div[data-testid="stRadio"] label[data-checked="true"] { background: linear-gradient(90deg, #b03a2e, #1a4b8c); border-color: transparent; }

  /* ---- transparent top header so the gradient shows through ---- */
  header[data-testid="stHeader"] { background: transparent !important; }

  /* ---- sidebar (targets both old and new Streamlit DOM) ---- */
  section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
      background: rgba(8,11,20,0.92) !important; border-right: 1px solid rgba(255,255,255,0.08);
  }
  section[data-testid="stSidebar"] *, div[data-testid="stSidebar"] * { color: #cfd4e2 !important; }

  /* ---- misc widget polish ---- */
  .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div, .stDateInput input {
      background: rgba(255,255,255,0.06) !important; color: #f1f2f6 !important; border-radius: 8px !important;
      border: 1px solid rgba(255,255,255,0.1) !important;
  }
  .stDataFrame { border-radius: 10px; overflow: hidden; }
  hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 Agentic AI Data Scientist</h1>
  <p>Upload a dataset, describe what you want in plain English, and specialized agents will
     clean it, explore it, query it, model it, and write up a report — with a full reasoning trace.</p>
</div>
""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                             help="If provided, agents use live Claude reasoning for judgment calls "
                                  "(target detection, task framing). Otherwise deterministic heuristics are used.")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    st.markdown("---")
    st.markdown("**Pipeline stages**")
    for stage in ["Data Understanding", "Cleaning", "Feature Engineering",
                  "EDA", "SQL (optional)", "Model Selection & Training", "Report"]:
        st.markdown(f"- {stage}")

# ---------------------------------------------------------------- INPUT
st.markdown('<div class="glass">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json", "parquet"])
    use_sample = st.checkbox("Use bundled sample dataset (customer churn)", value=not bool(uploaded))

with col2:
    instruction = st.text_area(
        "Instruction",
        placeholder="e.g. Predict customer churn. Focus on customers with more than 3 support calls.",
        height=110,
    )
    sql_query = st.text_input("Optional SQL query (runs against your data as a table named `data`)",
                               placeholder="SELECT plan_type, COUNT(*) FROM data GROUP BY plan_type;")

run = st.button("🚀 Run agent pipeline", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- EXECUTION
if run:
    dataset_path = None
    tmpdir = tempfile.mkdtemp()
    try:
        if uploaded is not None:
            dataset_path = os.path.join(tmpdir, uploaded.name)
            with open(dataset_path, "wb") as f:
                f.write(uploaded.getbuffer())
        elif use_sample:
            dataset_path = "sample_customer_churn.csv"
        else:
            st.warning("Upload a dataset or check the sample-dataset box.")
            st.stop()

        with st.spinner("Agents are working through the pipeline..."):
            orch = Orchestrator()
            queries = [sql_query] if sql_query.strip() else None
            result = orch.run(dataset_path, instruction=instruction, sql_queries=queries)
            st.session_state.result = result
        st.success("Pipeline complete.")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")

# ---------------------------------------------------------------- RESULTS
result = st.session_state.result
if result:
    ctx = result["context"]
    profile = ctx["profile"]
    target = ctx.get("target")
    model = ctx.get("model")

    st.markdown("## Overview")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">Rows</div><div class="value">{profile['n_rows']}</div></div>
      <div class="metric-card"><div class="label">Columns</div><div class="value">{profile['n_cols']}</div></div>
      <div class="metric-card"><div class="label">Target Column</div><div class="value">{target or '—'}</div></div>
      <div class="metric-card"><div class="label">Best Model</div><div class="value">{model['best_model'] if model else '—'}</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab_names = ["📋 Data Profile", "🧹 Cleaning", "📊 EDA", "🗄️ SQL", "🤖 Modeling",
                 "🔮 Predict New Data", "🧠 Agent Trace", "📄 Full Report"]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tab_names[0]
    active = st.radio("Section", tab_names, horizontal=True, key="active_tab", label_visibility="collapsed")
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if active == tab_names[0]:
        st.dataframe(pd.DataFrame([{
            "column": c["name"], "dtype": c["dtype"], "kind": c["kind"],
            "% missing": c["pct_missing"], "unique": c["n_unique"],
        } for c in profile["columns"]]), use_container_width=True)

    elif active == tab_names[1]:
        for line in ctx["clean_log"]:
            st.markdown(f"- {line}")

    elif active == tab_names[2]:
        eda = ctx["eda"]
        for insight in eda["insights"]:
            st.info(insight)
        for name, b64 in eda["plots"].items():
            if b64:
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                st.image(f"data:image/png;base64,{b64}")

    elif active == tab_names[3]:
        if ctx.get("sql"):
            for q, rows in ctx["sql"]["queries"].items():
                st.code(q, language="sql")
                if isinstance(rows, dict) and "error" in rows:
                    st.error(rows["error"])
                else:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("No SQL query was run for this pipeline execution.")

    elif active == tab_names[4]:
        if model:
            st.dataframe(pd.DataFrame(model["results"]), use_container_width=True)
            if model.get("confusion_matrix_plot"):
                st.image(f"data:image/png;base64,{model['confusion_matrix_plot']}")
        else:
            st.caption("No supervised target was detected/requested — modeling stage was skipped.")

    elif active == tab_names[5]:
        if model and model.get("best_pipeline") is not None:
            st.markdown(f"**Trained model: {model['best_model']}** — enter values for a new record "
                        f"and get a live prediction.")
            feature_cols = model["feature_columns"]
            df_for_ranges = result["cleaned_df"]
            input_row = {}

            # Group engineered date parts (e.g. signup_date_year/_month/_dayofweek)
            # back into a single date picker instead of three raw number fields.
            date_groups = {}   # base_name -> {"year": col, "month": col, "dayofweek": col}
            plain_cols = []
            for col in feature_cols:
                matched = False
                for suffix in ("_year", "_month", "_dayofweek"):
                    if col.endswith(suffix):
                        base = col[: -len(suffix)]
                        date_groups.setdefault(base, {})[suffix[1:]] = col
                        matched = True
                        break
                if not matched:
                    plain_cols.append(col)

            cols = st.columns(3)
            slot = 0
            for base, parts in date_groups.items():
                if {"year", "month", "dayofweek"} <= parts.keys():
                    with cols[slot % 3]:
                        import datetime as _dt
                        picked = st.date_input(base.replace("_", " ").title(), value=_dt.date.today())
                    input_row[parts["year"]] = picked.year
                    input_row[parts["month"]] = picked.month
                    input_row[parts["dayofweek"]] = picked.weekday()
                    slot += 1
                else:
                    # incomplete triplet — fall back to plain numeric fields
                    plain_cols.extend(parts.values())

            for col in plain_cols:
                with cols[slot % 3]:
                    if col in df_for_ranges.columns and pd.api.types.is_numeric_dtype(df_for_ranges[col]):
                        series = df_for_ranges[col].dropna()
                        # Show whole numbers for prediction inputs (age, calls, income, etc.)
                        # instead of decimal points — cleaner to type and reason about.
                        default_int = int(round(series.median())) if len(series) else 0
                        input_row[col] = st.number_input(col, value=default_int, step=1, format="%d")
                    elif col in df_for_ranges.columns:
                        options = sorted(df_for_ranges[col].dropna().unique().tolist())
                        input_row[col] = st.selectbox(col, options)
                    else:
                        input_row[col] = st.text_input(col, value="0")
                slot += 1

            if st.button("🔮 Predict"):
                import tools as _tools
                bundle = {
                    "pipeline": model["best_pipeline"],
                    "label_map": model["label_map"],
                    "feature_columns": model["feature_columns"],
                    "task": model["task"],
                }
                pred = _tools.predict_new(bundle, input_row)
                if pred["task"] == "classification":
                    if "probabilities" in pred:
                        probs = pred["probabilities"]
                        # Try to surface a plain-English verdict for binary yes/no-style targets
                        top_label = max(probs, key=probs.get)
                        top_pct = probs[top_label] * 100
                        positive_labels = {"1", "yes", "true", "churn", "churned"}
                        if str(top_label).lower() in positive_labels:
                            st.error(f"⚠️ Likely to leave — **{top_pct:.1f}%** predicted probability")
                        elif len(probs) == 2:
                            st.success(f"✅ Likely to stay — **{top_pct:.1f}%** predicted probability")
                        else:
                            st.success(f"Predicted class: **{top_label}** ({top_pct:.1f}%)")

                        chart_df = pd.DataFrame({
                            "class": [str(k) for k in probs.keys()],
                            "probability (%)": [round(v * 100, 1) for v in probs.values()],
                        }).set_index("class")
                        st.bar_chart(chart_df)
                        st.caption("Class labels are the raw encoded values from your target column "
                                   "(e.g. 0/1 for a binary flag) unless it already had text labels like Yes/No.")
                    else:
                        st.success(f"Predicted class: **{pred['prediction']}**")
                else:
                    st.success(f"Predicted value: **{pred['prediction']}**")

            if result.get("model_path"):
                with open(result["model_path"], "rb") as f:
                    st.download_button("⬇️ Download trained model (.joblib)", data=f.read(),
                                        file_name="trained_model.joblib")
                st.caption("Load it elsewhere with: `bundle = joblib.load('trained_model.joblib')` "
                           "then `tools.predict_new(bundle, {...})`")
        else:
            st.caption("No trained model available — modeling stage was skipped for this run.")

    elif active == tab_names[6]:
        for agent in ctx["agents"]:
            st.markdown(f"<div class='agent-card'><h4>{agent.name}</h4>" +
                        "".join(f"<div>• {s}</div>" for s in agent.trace) +
                        "</div>", unsafe_allow_html=True)

    elif active == tab_names[7]:
        st.download_button("⬇️ Download HTML report", data=result["html_report"],
                            file_name="agentic_ds_report.html", mime="text/html")
        st.components.v1.html(result["html_report"], height=800, scrolling=True)

    st.markdown('</div>', unsafe_allow_html=True)
