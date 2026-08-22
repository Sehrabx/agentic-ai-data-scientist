"""
predict.py — Load a model trained by main.py/app.py and predict on new data.

Usage:
    python predict.py --model outputs/trained_model.joblib \\
        --input '{"age": 45, "monthly_income": 52000, "tenure_years": 1.2, "plan_type": "Basic", "support_calls": 5}'

Or batch-predict a CSV of new records (must have the same feature columns as training):
    python predict.py --model outputs/trained_model.joblib --csv new_customers.csv --out predictions.csv
"""
import argparse
import json
import joblib
import pandas as pd

import tools


def main():
    p = argparse.ArgumentParser(description="Predict with a saved Agentic AI Data Scientist model")
    p.add_argument("--model", required=True, help="Path to .joblib model bundle")
    p.add_argument("--input", help="Single JSON record, e.g. '{\"age\": 45, ...}'")
    p.add_argument("--csv", help="CSV of multiple records to predict in batch")
    p.add_argument("--out", default="predictions.csv", help="Where to write batch predictions")
    args = p.parse_args()

    bundle = joblib.load(args.model)
    print(f"Loaded {bundle['model_name']} ({bundle['task']}) — expects features: {bundle['feature_columns']}")

    if args.input:
        row = json.loads(args.input)
        result = tools.predict_new(bundle, row)
        print(json.dumps(result, indent=2))

    elif args.csv:
        df = pd.read_csv(args.csv)
        rows = df[bundle["feature_columns"]].to_dict(orient="records")
        preds = [tools.predict_new(bundle, r) for r in rows]
        df["prediction"] = [p["prediction"] for p in preds]
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} predictions to {args.out}")

    else:
        print("Provide either --input '<json>' or --csv <file>. See --help.")


if __name__ == "__main__":
    main()
