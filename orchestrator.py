"""
orchestrator.py — Coordinates the agent pipeline.

Given a dataset path + a natural-language instruction, the orchestrator:
  1. Loads the data
  2. Runs DataUnderstandingAgent -> profile + target
  3. Runs CleaningAgent -> cleaned df
  4. Runs FeatureEngineeringAgent (if a target/model task exists)
  5. Runs EDAAgent -> plots + insights
  6. Runs SQLAgent if the instruction contains SQL-like requests
  7. Runs ModelSelectionTrainingAgent if a target is present
  8. Runs ReportAgent -> final HTML

The orchestrator itself contains no ML/statistics logic — it just wires
agents together and decides which stages are relevant to the instruction.
"""
from __future__ import annotations
import re
import pandas as pd

import tools
from llm import Reasoner
from agents import (
    DataUnderstandingAgent, CleaningAgent, EDAAgent, SQLAgent,
    FeatureEngineeringAgent, ModelSelectionTrainingAgent, ReportAgent,
)


class Orchestrator:
    def __init__(self, live_llm: bool | None = None):
        self.reasoner = Reasoner(live=live_llm)

    def run(self, dataset_path: str, instruction: str = "", sql_queries: list[str] | None = None) -> dict:
        df = tools.load_dataset(dataset_path)

        agents_run = []

        du = DataUnderstandingAgent(self.reasoner)
        du_out = du.run(df, instruction)
        agents_run.append(du)
        target = du_out["target"]

        clean = CleaningAgent(self.reasoner)
        clean_out = clean.run(df, target)
        agents_run.append(clean)
        df_clean = clean_out["df"]

        task = None
        if target and target in df_clean.columns:
            task = tools.infer_task(df_clean, target)

        fe = FeatureEngineeringAgent(self.reasoner)
        if target:
            fe_out = fe.run(df_clean, target)
            df_features = fe_out["df"]
        else:
            fe.log("No target column — skipping supervised feature engineering.")
            df_features = df_clean
        agents_run.append(fe)

        eda = EDAAgent(self.reasoner)
        eda_out = eda.run(df_features, target, task)
        agents_run.append(eda)

        sql_out = None
        auto_queries = sql_queries or self._extract_sql(instruction)
        if auto_queries:
            sql_agent = SQLAgent(self.reasoner)
            sql_out = sql_agent.run(df_features, auto_queries)
            agents_run.append(sql_agent)

        model_out = None
        wants_model = target is not None and (
            not instruction or any(k in instruction.lower() for k in
                ["predict", "model", "classif", "regress", "train", "accuracy"])
        )
        if wants_model:
            model_agent = ModelSelectionTrainingAgent(self.reasoner)
            model_out = model_agent.run(df_features, target)
            agents_run.append(model_agent)

        report_agent = ReportAgent(self.reasoner)
        context = {
            "profile": du_out["profile"],
            "target": target,
            "clean_log": clean_out["log"],
            "eda": eda_out,
            "model": model_out,
            "sql": sql_out,
            "agents": agents_run,
            "instruction": instruction,
        }
        html = report_agent.run(context)
        agents_run.append(report_agent)

        model_path = None
        if model_out and model_out.get("best_pipeline") is not None:
            import os
            os.makedirs("outputs", exist_ok=True)
            model_path = tools.save_model(model_out, "outputs/trained_model.joblib")

        return {
            "html_report": html,
            "cleaned_df": df_features,
            "context": context,
            "model_path": model_path,
        }

    @staticmethod
    def _extract_sql(instruction: str) -> list[str]:
        if not instruction:
            return []
        matches = re.findall(r"(SELECT .*?;)", instruction, flags=re.IGNORECASE | re.DOTALL)
        return [m.strip() for m in matches]
