"""
main.py — CLI entry point.

Usage:
    python main.py --data path/to/data.csv --instruction "Predict churn" --out report.html

If ANTHROPIC_API_KEY is set in the environment, agents use live Claude
reasoning for judgment calls (target detection, task framing). Otherwise
they use the built-in deterministic heuristics — the pipeline still runs
fully end-to-end either way.
"""
import argparse
from orchestrator import Orchestrator


def main():
    p = argparse.ArgumentParser(description="Agentic AI Data Scientist")
    p.add_argument("--data", required=True, help="Path to CSV/XLSX/JSON/Parquet dataset")
    p.add_argument("--instruction", default="", help="Natural language instruction, e.g. 'Predict the target column'")
    p.add_argument("--out", default="outputs/report.html", help="Output HTML report path")
    p.add_argument("--sql", action="append", default=None, help="Optional SQL query to run (repeatable)")
    args = p.parse_args()

    orch = Orchestrator()
    result = orch.run(args.data, instruction=args.instruction, sql_queries=args.sql)

    with open(args.out, "w") as f:
        f.write(result["html_report"])
    print(f"Report written to {args.out}")
    if result.get("model_path"):
        print(f"Trained model saved to {result['model_path']}")
        print("Use it later with:  python predict.py --model outputs/trained_model.joblib --input '{...}'")

    for agent in result["context"]["agents"]:
        print(f"\n[{agent.name}]")
        for step in agent.trace:
            print(f"  - {step}")


if __name__ == "__main__":
    main()
