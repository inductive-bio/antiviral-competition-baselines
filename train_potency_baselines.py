from baseline_models import get_fingerprint_baseline, get_descriptor_baseline
from evaluation.potency import evaluate_potency_predictions
import pandas as pd
from rdkit import Chem
import numpy as np
import polaris as po


def load_dataset():
    dataset = po.load_dataset("asap-discovery/antiviral-potency-2025-unblinded")
    rows = [dataset[i] for i in dataset.rows]
    df_labels = pd.DataFrame(rows)
    df_labels["ROMol"] = df_labels["CXSMILES"].apply(Chem.MolFromSmiles)
    df_train = df_labels.query("Set == 'Train'").reset_index(drop=True)
    df_test = df_labels.query("Set == 'Test'").reset_index(drop=True)
    return df_train, df_test


def train_baseline_model(df_train, df_test, model_generator):
    preds = {}
    for c in ["pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)"]:
        print(f" - Training {c}...")
        df_train_target = df_train.query(f"`{c}`.notna()").reset_index(drop=True)
        sklearn_model = model_generator()
        sklearn_model.fit(df_train_target["ROMol"].values, df_train_target[c].values)
        prop_pred = sklearn_model.predict(df_test["ROMol"].values)
        preds[c] = prop_pred
    return preds


if __name__ == "__main__":
    df_train, df_test = load_dataset()
    for model_name, model_generator in [
        ("fingerprint_baseline", get_fingerprint_baseline),
        ("descriptor_baseline", get_descriptor_baseline),
    ]:
        print(f"Training {model_name}...")
        preds = train_baseline_model(df_train, df_test, model_generator)
        pd.DataFrame(preds).to_csv(
            f"data_out/potency_preds_{model_name}.csv", index=False
        )
        print(f"Evaluating {model_name}...")
        metrics = evaluate_potency_predictions(
            df_test.to_dict(orient="list"), preds, model_name
        )
        metrics.to_csv(f"data_out/potency_metrics_{model_name}.csv", index=False)
        summary_metrics = (
            metrics.groupby(["Target Label", "Metric"])["Score"]
            .agg(["mean", "std"])
            .reset_index()
            .query("`Target Label` == 'aggregated'")
        )
        print(f"Summary metrics for {model_name}:")
        print(summary_metrics)
