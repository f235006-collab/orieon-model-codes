"""Compare newly generated metrics with the saved paper reference metrics."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACTUAL_PATH = ROOT / "results" / "metrics.csv"
EXPECTED_PATH = ROOT / "expected_results" / "metrics.csv"
EXPECTED_PROTOCOL = ROOT / "expected_results" / "data_protocol.json"
ACTUAL_PROTOCOL = ROOT / "results" / "data_protocol.json"

actual = pd.read_csv(ACTUAL_PATH)
expected = pd.read_csv(EXPECTED_PATH)
keys = ["Model", "Split"]
missing = expected.merge(actual[keys], on=keys, how="left", indicator=True)
missing = missing[missing["_merge"] == "left_only"]
if not missing.empty:
    print("Missing model/split results:")
    print(missing[keys].to_string(index=False))
    raise SystemExit(1)

merged = expected.merge(actual, on=keys, suffixes=("_paper", "_run"), validate="one_to_one")
numeric = ["MAE", "RMSE", "R2", "MAPE_pct", "Directional_Accuracy_pct", "n"]
for column in numeric:
    merged[f"{column}_abs_diff"] = (
        merged[f"{column}_run"] - merged[f"{column}_paper"]
    ).abs()

print("\nHeld-out test RMSE: new run versus paper reference")
test = merged[merged["Split"] == "test"].copy()
test["RMSE_difference"] = test["RMSE_run"] - test["RMSE_paper"]
print(
    test[["Model", "RMSE_run", "RMSE_paper", "RMSE_difference"]]
    .sort_values("RMSE_run")
    .to_string(index=False, float_format=lambda value: f"{value:,.6f}")
)

print("\nMaximum absolute differences across all model/split rows:")
for column in numeric:
    print(f"  {column}: {merged[f'{column}_abs_diff'].max():.12g}")

protocol_matches = ACTUAL_PROTOCOL.exists() and ACTUAL_PROTOCOL.read_bytes() == EXPECTED_PROTOCOL.read_bytes()
print(f"\nData protocol byte-identical to reference: {protocol_matches}")
if not protocol_matches and ACTUAL_PROTOCOL.exists():
    import json

    current_protocol = json.loads(ACTUAL_PROTOCOL.read_text(encoding="utf-8"))
    reference_protocol = json.loads(EXPECTED_PROTOCOL.read_text(encoding="utf-8"))
    ignored = {"features"}
    same = all(current_protocol.get(key) == reference_protocol.get(key) for key in reference_protocol if key not in ignored)
    same = same and current_protocol.get("features") == reference_protocol.get("features")
    print(f"Data protocol semantically identical to reference: {same}")

all_close = all(
    np.allclose(
        merged[f"{column}_run"], merged[f"{column}_paper"], rtol=1e-8, atol=1e-8
    )
    for column in numeric
)
print(f"All reported metrics match within tight numerical tolerance: {all_close}")
print(
    "Note: neural-network results can differ with library versions or hardware; "
    "a difference is not automatically an execution error."
)
