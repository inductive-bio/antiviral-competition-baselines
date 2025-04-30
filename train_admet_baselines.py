from baseline_models import get_fingerprint_baseline, get_descriptor_baseline
from evaluation.admet import evaluate_admet_predictions
import pandas as pd
from rdkit import Chem
import numpy as np
import polaris as po

# some training measurements are at/near 0,
# leading to NAN or unreasonable log-transformed values.
# Clip these to a reasonable minimum value.
train_clip_vals = {
    "HLM": 5,
    "MLM": 5,
    "MDR1-MDCKII": 0.05,
}


def load_dataset():
    dataset = po.load_dataset("asap-discovery/antiviral-admet-2025-unblinded")
    rows = [dataset[i] for i in dataset.rows]
    df_labels = pd.DataFrame(rows)
    df_labels["ROMol"] = df_labels["CXSMILES"].apply(Chem.MolFromSmiles)
    df_train = df_labels.query("Set == 'Train'").reset_index(drop=True)
    df_test = df_labels.query("Set == 'Test'").reset_index(drop=True)
    return df_train, df_test


def train_baseline_model(df_train, df_test, model_generator):
    preds = {}
    for c in ["HLM", "KSOL", "LogD", "MDR1-MDCKII", "MLM"]:
        print(f" - Training {c}...")
        df_train_target = df_train.query(f"`{c}`.notna()").reset_index(drop=True)
        if c in train_clip_vals:
            df_train_target[c] = df_train_target[c].clip(lower=train_clip_vals[c])
        if c != "LogD":
            df_train_target[c] = np.log10(df_train_target[c])
        sklearn_model = model_generator()
        sklearn_model.fit(df_train_target["ROMol"].values, df_train_target[c].values)
        prop_pred = sklearn_model.predict(df_test["ROMol"].values)
        if c != "LogD":
            prop_pred = 10**prop_pred
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
            f"data_out/admet_preds_{model_name}.csv", index=False
        )
        print(f"Evaluating {model_name}...")
        metrics = evaluate_admet_predictions(
            df_test.to_dict(orient="list"), preds, model_name
        )
        metrics.to_csv(f"data_out/admet_metrics_{model_name}.csv", index=False)
        summary_metrics = (
            metrics.groupby(["Target Label", "Metric"])["Score"]
            .agg(["mean", "std"])
            .reset_index()
            .query("`Target Label` == 'aggregated'")
        )
        print(f"Summary metrics for {model_name}:")
        print(summary_metrics)
