"""
agents.py — Specialized agents, each responsible for one stage of the
data science workflow. Each agent:
  1. reasons about *what to do* given context (target column? which task?
     which models?) — via self.reasoner (LLM if available, else heuristic)
  2. executes deterministic work via tools.py
  3. returns a structured result + a natural-language note for the report

This is the core "agentic" pattern: reasoning and execution are separate,
traceable steps, not one opaque generation.
"""
from __future__ import annotations
import pandas as pd

import tools
from llm import Reasoner


class BaseAgent:
    name = "base"

    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner
        self.trace: list[str] = []

    def log(self, msg: str):
        self.trace.append(msg)


class DataUnderstandingAgent(BaseAgent):
    name = "DataUnderstandingAgent"

    def run(self, df: pd.DataFrame, instruction: str) -> dict:
        profile = tools.profile_dataset(df)
        self.log(f"Profiled dataset: {profile['n_rows']} rows × {profile['n_cols']} columns.")

        target = self._infer_target(df, instruction, profile)
        self.log(f"Selected target column: '{target}'" if target else "No target column identified — running in unsupervised/EDA-only mode.")
        return {"profile": profile, "target": target}

    def _infer_target(self, df: pd.DataFrame, instruction: str, profile: dict) -> str | None:
        # 1. explicit mention in the instruction
        instr = instruction.lower()
        for col in df.columns:
            if col.lower() in instr and ("predict" in instr or "target" in instr or "classify" in instr or "estimate" in instr):
                return col
        # 2. common naming conventions
        common_names = ["target", "label", "class", "outcome", "y", "price", "survived", "churn", "diagnosis"]
        for cand in common_names:
            for col in df.columns:
                if col.lower() == cand:
                    return col
        # 3. no strong signal -> no supervised target
        if "predict" in instr or "classify" in instr or "regression" in instr or "model" in instr:
            # fall back to last column, a common dataset convention
            return df.columns[-1]
        return None


class CleaningAgent(BaseAgent):
    name = "CleaningAgent"

    def run(self, df: pd.DataFrame, target: str | None) -> dict:
        cleaned, log = tools.clean_dataset(df, target=target)
        for entry in log:
            self.log(entry)
        return {"df": cleaned, "log": log}


class EDAAgent(BaseAgent):
    name = "EDAAgent"

    def run(self, df: pd.DataFrame, target: str | None, task: str | None) -> dict:
        summary = tools.eda_summary(df, target)
        plots = {
            "missingness": tools.plot_missingness(df),
            "correlation_heatmap": tools.plot_correlation_heatmap(df),
            "distributions": tools.plot_distributions(df),
        }
        if target and task:
            plots["target_relationship"] = tools.plot_target_relationship(df, target, task)

        insights = []
        if summary.get("target_relationship"):
            top = list(summary["target_relationship"].items())[:3]
            insights.append(
                "Features most correlated with the target: "
                + ", ".join(f"{k} ({v})" for k, v in top)
            )
        if df.isna().sum().sum() == 0:
            insights.append("No missing values remain after cleaning.")
        self.log(f"Generated {sum(1 for p in plots.values() if p)} exploratory plots.")
        for i in insights:
            self.log(i)
        return {"summary": summary, "plots": plots, "insights": insights}


class SQLAgent(BaseAgent):
    name = "SQLAgent"

    def run(self, df: pd.DataFrame, queries: list[str] | None) -> dict:
        results = {}
        for q in (queries or []):
            try:
                results[q] = tools.run_sql(df, q).to_dict(orient="records")
                self.log(f"Executed SQL query: {q}")
            except Exception as e:
                results[q] = {"error": str(e)}
                self.log(f"SQL query failed: {q} ({e})")
        return {"queries": results}


class FeatureEngineeringAgent(BaseAgent):
    name = "FeatureEngineeringAgent"

    def run(self, df: pd.DataFrame, target: str) -> dict:
        before_cols = set(df.columns)
        engineered = tools.engineer_features(df, target)
        new_cols = set(engineered.columns) - before_cols
        if new_cols:
            self.log(f"Engineered {len(new_cols)} new features from datetime/text columns: {sorted(new_cols)}")
        else:
            self.log("No datetime/text columns detected for feature extraction; passing data through.")
        return {"df": engineered}


class ModelSelectionTrainingAgent(BaseAgent):
    name = "ModelSelectionTrainingAgent"

    def run(self, df: pd.DataFrame, target: str) -> dict:
        task = tools.infer_task(df, target)
        self.log(f"Inferred task type: {task} (based on target cardinality/dtype).")
        result = tools.train_and_evaluate(df, target, task)
        self.log(f"Trained {len(result['results'])} candidate models on {result['n_train']} rows, "
                  f"evaluated on {result['n_test']} held-out rows.")
        self.log(f"Best model selected: {result['best_model']}")
        return result


class ReportAgent(BaseAgent):
    name = "ReportAgent"

    def run(self, context: dict) -> str:
        from report import build_html_report
        html = build_html_report(context)
        self.log("Compiled final HTML report.")
        return html
