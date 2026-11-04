# ORIEON — Reproducible Gold-Forecasting Model Codes

Yeh package paper mein use hone wali **complete methodology** ko reproduce karta hai: data preparation, sentiment filtering/aggregation, chronological splits, statistical models, conventional machine learning, neural baselines, dynamic graph hybrid, ablations, evaluation, aur figures.

**Verification:** complete packaged workflow supplied CSV files par dobara run kiya gaya. Tamam 22 model/split metric rows, data protocol, predictions, aur five figures successfully generate hue; metrics reference results se exactly match kiye in the pinned environment.

Dono required input files `data/` folder mein pehle se mojood hain:

- `data/gold_clean.csv`
- `data/news_with_scores.csv`

`news_with_scores.csv` bhi zaroori hai kyunke paper ka full model financial-news sentiment use karta hai. Agar aap apni updated files use karein to unke filenames aur columns same rakhne honge; nateeja tab paper se mukhtalif ho sakta hai.

## 1. Quick start — Roman Urdu

Python **3.10–3.12** recommended hai. Terminal/Command Prompt ko is folder ke andar open karein, phir:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` tamam stages ko sahi tarteeb mein chalata hai. Neural aur graph models CPU par train hote hain; total runtime aap ke processor par depend karega aur kaafi lamba ho sakta hai. Program ko beech mein band na karein jab tak final completion message na aa jaye.

Results yahan milenge:

- **Tamam metrics:** `results/metrics.csv`
- **Validation/test predictions:** `results/*_val_predictions.csv` aur `results/*_test_predictions.csv`
- **Best neural checkpoints:** `results/*_best.pt`
- **Training histories:** `results/*_training.json`, `results/deep_training.json`
- **Selected models/hyperparameters:** `results/selected_classical.json`, `results/selected_temporal.json`, `results/classical_tuning.json`
- **Figures:** `figures/`
- **Paper ke reference outputs:** `expected_results/`

## 2. Models implemented

| File | Models / purpose |
|---|---|
| `common.py` | News relevance filter, daily sentiment aggregation, 33 model features, 70/15/15 chronological split, metrics, prediction saving |
| `classical_experiments.py` | Persistence, rolling one-step ARIMA, Ridge, random forest, LightGBM |
| `deep_baselines.py` | MLP, LSTM, GRU; 20-day sequences; validation-based temporal-model selection |
| `dynamic_experiment.py` | Dynamic GNN–temporal hybrid, static-graph ablation, no-sentiment ablation |
| `generate_figures.py` | Five paper figures and derived comparison statistics |
| `compare_results.py` | New metrics versus the paper's saved reference metrics |
| `run_all.py` | Ordered execution of the workflow |

The target is `y_close_t1` (next-day gold closing price). `y_close_t1` and `y_logret` are excluded from model inputs. Feature and target scalers are fitted on training rows only. Model and hyperparameter selection use validation performance, not test performance.

## 3. Run stages separately

Agar poora workflow ek saath nahi chalana to:

```bash
python run_all.py classical
python run_all.py deep
python run_all.py dynamic
python run_all.py figures
python run_all.py compare
```

Graph experiments ko alag bhi run kar saktay hain (pehle `classical` aur `deep` stages zaroor complete honi chahiye):

```bash
python run_all.py dynamic       # dynamic + static + no-sentiment
python run_all.py static        # static graph only
python run_all.py nosentiment   # dynamic graph without sentiment only
```

Individual scripts bhi chal saktay hain, lekin `run_all.py` prerequisites check karta hai aur recommended route hai.

## 4. Expected test RMSE from the supplied reference run

| Model | Test RMSE |
|---|---:|
| Persistence | 55.055 |
| Ridge | 55.301 |
| ARIMA(1,1,0) | 76.555 |
| MLP | 315.990 |
| Static GNN–GRU | 1,418.394 |
| Dynamic GNN–GRU | 1,433.044 |
| Dynamic GNN–GRU, no sentiment | 1,438.927 |
| GRU | 1,549.719 |
| LSTM | 1,592.510 |
| Random forest | 1,635.167 |
| LightGBM | 1,654.442 |

Poori table (MAE, RMSE, R², MAPE, directional accuracy, validation aur test) `expected_results/metrics.csv` mein hai.

**Important:** complex models ka weak test performance code error nahi hai. Supplied sample ke test period mein pronounced upward regime shift hai; paper is limitation ko report karta hai aur dynamic graph ko best model claim nahi karta.

## 5. Reproducibility notes

- Fixed seed: **42**; sequence lookback: **20 days**.
- Chronological partition: 3,523 training, 755 validation, 756 test rows for the supplied data.
- Neural training: maximum 60 epochs, early-stopping patience 10, batch size 64.
- Exact reference dependency versions `requirements.txt` mein pinned hain.
- Different operating systems, BLAS implementations, PyTorch/library versions, ya hardware ki wajah se khaas taur par neural results mein small numerical differences aa saktay hain. Is liye final comparison report differences dikhati hai, unhen chhupati nahi.
- Kisi naye dataset par paper wali numerical values expect na karein; woh values sirf supplied files aur stated protocol se belong karti hain.
