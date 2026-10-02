"""Batch1에서 변수·회귀 모델을 선정하고 고정한 모델을 Batch2에서 평가한다."""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
KEYS = ["batch", "cell_id"]
FEATURES = ["log10_dqv_var", "equivalent_c_rate", "thermal_charge_interaction"]
SEED = 42
TARGET_MAPE = 9.1


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert len(y) == len(prediction) and np.isfinite(prediction).all() and (y > 0).all()
    return {
        "MAPE_pct": float(100 * mean_absolute_percentage_error(y, prediction)),
        "MAE_cycles": float(mean_absolute_error(y, prediction)),
        "RMSE_cycles": float(root_mean_squared_error(y, prediction)),
        "bias_cycles": float(np.mean(prediction - y)),
    }


def select_model(output_dir=ROOT / "outputs"):
    """Batch1만 읽고 Hold-out 점수는 선정에 사용하지 않는다."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    batch1 = pd.read_csv(DATA / "batch1_model_candidates.csv")
    policies = pd.read_csv(DATA / "batch1_cells.csv")
    assert set(batch1["batch"]) == {"batch1"} and len(batch1) == 46
    assert not batch1.duplicated(KEYS).any() and not policies.duplicated("cell_id").any()
    batch1 = batch1.merge(policies[["cell_id", "policy"]], on="cell_id", validate="one_to_one")
    batch1 = batch1.sort_values(KEYS).reset_index(drop=True)
    policy_values = batch1["policy"].str.extract(r"^([\d.]+)C\((\d+)%\)-([\d.]+)C")
    assert policy_values.notna().all().all()
    batch1["protocol"] = policy_values.astype(float).astype(str).agg("|".join, axis=1)
    assert np.isfinite(batch1[FEATURES + ["cycle_life"]].to_numpy()).all()
    assert batch1["cycle_life"].gt(100).all()
    train_idx, valid_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    ).split(batch1, groups=batch1["protocol"]))
    train = batch1.iloc[train_idx].reset_index(drop=True)
    valid = batch1.iloc[valid_idx].reset_index(drop=True)
    assert len(train) == 35 and len(valid) == 11
    assert set(train["protocol"]).isdisjoint(valid["protocol"])
    assert set(train["cell_id"]).isdisjoint(valid["cell_id"])
    cv = list(GroupKFold(5).split(train, groups=train["protocol"]))
    coverage = np.zeros(len(train), dtype=int)
    for fit_idx, check_idx in cv:
        assert set(train.iloc[fit_idx]["protocol"]).isdisjoint(train.iloc[check_idx]["protocol"])
        coverage[check_idx] += 1
    assert np.all(coverage == 1)

    feature_sets = {"L": FEATURES[:1], "L+E": FEATURES[:2],
                    "L+T": [FEATURES[0], FEATURES[2]], "L+E+T": FEATURES}
    model_space = [
        ("LinearRegression", LinearRegression(), {}, 0),
        ("Ridge", Ridge(), {"regressor__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}, 1),
        ("DecisionTree", DecisionTreeRegressor(random_state=SEED),
         {"regressor__max_depth": [2, 3, 4], "regressor__min_samples_leaf": [3, 5]}, 2),
        ("SVR", SVR(kernel="rbf"), {"regressor__C": [100, 1000, 10000],
         "regressor__epsilon": [10, 50], "regressor__gamma": ["scale", 0.1]}, 3),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
         {"regressor__max_depth": [2, 4, None], "regressor__min_samples_leaf": [2, 4]}, 4),
    ]
    rows, fitted = [], {}
    for feature_set, columns in feature_sets.items():
        for name, regressor, grid, rank in model_space:
            search = GridSearchCV(
                Pipeline([("scale", StandardScaler()), ("regressor", regressor)]),
                grid, cv=cv, scoring="neg_mean_absolute_percentage_error",
                n_jobs=1, error_score="raise",
            ).fit(train[columns], train["cycle_life"])
            scores = [-100 * search.cv_results_[f"split{i}_test_score"][search.best_index_] for i in range(5)]
            candidate = f"{name} ({feature_set})"
            rows.append({
                "candidate": candidate, "model": name, "feature_set": feature_set,
                "features": list(columns), "n_features": len(columns), "model_rank": rank,
                "CV_MAPE_pct": float(np.mean(scores)), "CV_std_pp": float(np.std(scores)),
                "CV_se_pp": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
                "fold_MAPE_pct": [float(s) for s in scores], "params": search.best_params_,
            })
            fitted[candidate] = search.best_estimator_
    comparison = pd.DataFrame(rows)
    best = comparison.loc[comparison["CV_MAPE_pct"].idxmin()]
    threshold = float(best["CV_MAPE_pct"] + best["CV_se_pp"])
    comparison["within_one_se"] = comparison["CV_MAPE_pct"].le(threshold)
    eligible = comparison.loc[comparison["within_one_se"]].sort_values(
        ["n_features", "model_rank", "CV_std_pp", "CV_MAPE_pct"]
    )
    selected = eligible.iloc[0].to_dict()
    model = fitted[selected["candidate"]]
    columns = selected["features"]
    scaler = model.named_steps["scale"]
    assert scaler.n_samples_seen_ == len(train)
    assert np.allclose(scaler.mean_, train[columns].mean().to_numpy())
    # ponytail: CV는 후보·파라미터 선택 점수이며, 별도 Hold-out으로 고정 모델을 검증한다.
    selection = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "selection_batch": "batch1", "selection_n": 35, "holdout_n": 11,
        "target": "cycle_life (raw cycles)", "prediction_cycle": 100, "seed": SEED,
        "candidate_features": FEATURES, "selected": selected,
        "minimum_CV_candidate": best["candidate"], "minimum_CV_MAPE_pct": float(best["CV_MAPE_pct"]),
        "one_se_threshold_MAPE_pct": threshold,
        "selection_rule": "minimum CV MAPE + its sample fold standard error; then fewer features, model simplicity rank, lower fold SD, lower mean",
        "model_simplicity_order": [row[0] for row in model_space],
        "cv": "5-fold GroupKFold by numeric charging protocol",
        "holdout": "GroupShuffleSplit(test_size=0.2, random_state=42), not used for selection",
        "comparison": comparison.to_dict("records"),
        "training_rows": train[KEYS + ["policy", "protocol"]].to_dict("records"),
        "validation_rows": valid[KEYS + ["policy", "protocol"]].to_dict("records"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "joblib"]},
        "requirements_sha256": hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest(),
        "input_file_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                              for name in ["batch1_model_candidates.csv", "batch1_cells.csv"]},
    }
    selection_path = out / "model_selection_manifest.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    csv = comparison.copy()
    for column in ["features", "fold_MAPE_pct", "params"]:
        csv[column] = csv[column].map(lambda v: json.dumps(v, ensure_ascii=False))
    csv.to_csv(out / "model_comparison.csv", index=False)
    joblib.dump(model, out / "battery_model.joblib")
    print(comparison[["candidate", "CV_MAPE_pct", "CV_std_pp", "within_one_se"]].sort_values("CV_MAPE_pct").round(3))
    print(f"선택 모델: {selected['candidate']}; 1-SE 기준 {threshold:.3f}%")
    return {"model": model, "features": columns, "selection": selection,
            "batch1": batch1, "train": train, "valid": valid, "cv": cv, "out": out}


def batch2_features():
    """고정한 L·E·T 생성식을 평가 데이터에 적용한다."""
    cells = pd.read_csv(DATA / "batch2_cells.csv").assign(batch="batch2")
    known = cells.dropna(subset=["cycle_life"]).copy()
    assert not cells.duplicated(KEYS).any()
    policies = known["policy"].str.extract(r"^([\d.]+)C\((\d+)%\)-([\d.]+)C").astype(float)
    assert policies.notna().all().all()
    c1, s, c2 = policies[0], policies[1] / 100, policies[2]
    assert s.between(0, 0.8).all() and c1.gt(0).all() and c2.gt(0).all()
    frame = known[KEYS + ["policy", "cycle_life"]].copy()
    frame["protocol"] = policies.astype(str).agg("|".join, axis=1)
    frame["equivalent_c_rate"] = 0.8 / (s / c1 + (0.8 - s) / c2)
    summary = pd.read_csv(DATA / "batch2_cycle_summary.csv")
    measurements = ["QCharge", "QDischarge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]
    placeholder = summary["cycle"].eq(1) & summary[measurements].eq(0).all(axis=1)
    early = summary.loc[summary["cycle"].between(1, 100) & ~placeholder]
    temperature = early.groupby("cell_id")["Tavg"].mean()
    frame["thermal_charge_interaction"] = (
        frame["cell_id"].map(temperature) - 30
    ) * (c1 * s + c2 * (0.8 - s))
    voltage = pd.read_csv(DATA / "batch2_vdlin.csv")["Vdlin"].to_numpy()
    reference = pd.read_csv(DATA / "batch1_vdlin.csv")["Vdlin"].to_numpy()
    assert len(voltage) == 1000 and np.allclose(voltage, reference, rtol=0, atol=1e-12)
    logs = {}
    with np.load(DATA / "batch2_cycle_profiles.npz", allow_pickle=False) as archive:
        for cell_id in frame["cell_id"]:
            cycles = archive[f"cell_{cell_id}_cycle"]
            indices = [np.flatnonzero(cycles == c) for c in [10, 100]]
            assert all(len(i) == 1 for i in indices)
            q = archive[f"cell_{cell_id}_qdlin"]
            assert q.shape == (len(cycles), len(voltage))
            q10, q100 = q[indices[0][0]], q[indices[1][0]]
            variance = np.var(q100 - q10)
            assert np.isfinite([q10, q100]).all() and variance > 0
            logs[cell_id] = float(np.log10(variance))
            del q
    frame["log10_dqv_var"] = frame["cell_id"].map(logs)
    assert len(frame) == 39 and np.isfinite(frame[FEATURES + ["cycle_life"]].to_numpy()).all()
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    return frame, cells.loc[cells["cycle_life"].isna(), KEYS + ["policy"]].to_dict("records")


def evaluate_model(state):
    """선정한 고정 모델을 평가하고 평가 결과로 변수나 모델을 바꾸지 않는다."""
    model, columns, train, valid = state["model"], state["features"], state["train"], state["valid"]
    out, selection = state["out"], state["selection"]
    selection_bytes = (out / "model_selection_manifest.json").read_bytes()
    selected = selection["selected"]
    prediction_frames, folds = [], []
    for fold, (fit_idx, check_idx) in enumerate(state["cv"], 1):
        fit_data, check_data = train.iloc[fit_idx], train.iloc[check_idx]
        fold_model = clone(model).fit(fit_data[columns], fit_data["cycle_life"])
        assert fold_model.named_steps["scale"].n_samples_seen_ == len(fit_data)
        prediction = fold_model.predict(check_data[columns])
        folds.append({"fold": fold, "n": len(check_data), **metrics(check_data["cycle_life"], prediction)})
        prediction_frames.append(check_data.assign(split="train_cv", fold=fold, predicted_cycle_life=prediction))
    train_mape = float(np.mean([r["MAPE_pct"] for r in folds]))
    assert np.isclose(train_mape, selected["CV_MAPE_pct"])
    assert np.allclose([r["MAPE_pct"] for r in folds], selected["fold_MAPE_pct"])
    valid_prediction = model.predict(valid[columns])
    evaluations = {"valid": {"n": len(valid), **metrics(valid["cycle_life"], valid_prediction)}}
    prediction_frames.append(valid.assign(split="valid", fold=0, predicted_cycle_life=valid_prediction))
    # 변수·모델·파라미터의 선택을 저장한 뒤에만 Batch2를 읽는다.
    test, missing_labels = batch2_features()
    test_prediction = model.predict(test[columns])
    evaluations["test"] = {"n": len(test), **metrics(test["cycle_life"], test_prediction)}
    prediction_frames.append(test.assign(split="test", fold=0, predicted_cycle_life=test_prediction))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions[KEYS + ["policy", "protocol", *columns, "cycle_life",
                                     "split", "fold", "predicted_cycle_life"]]
    predictions = predictions.sort_values(["split", *KEYS]).reset_index(drop=True)
    predictions["error_cycles"] = predictions["predicted_cycle_life"] - predictions["cycle_life"]
    predictions["APE_pct"] = 100 * predictions["error_cycles"].abs() / predictions["cycle_life"]
    assert len(predictions) == 85 and not predictions.duplicated(KEYS).any()
    assert set(predictions["batch"]) == {"batch1", "batch2"}
    gaps = {"Train-Valid": evaluations["valid"]["MAPE_pct"] - train_mape,
            "Valid-Test": evaluations["test"]["MAPE_pct"] - evaluations["valid"]["MAPE_pct"],
            "Target-Test": evaluations["test"]["MAPE_pct"] - TARGET_MAPE}
    performance = pd.DataFrame([
        {"구분": "Train (Batch1 CV)", "MAPE (%)": train_mape, "비고": "학습 35개 셀, 프로토콜 5-fold 평균"},
        {"구분": "Valid (Batch1 Hold-out)", "MAPE (%)": evaluations["valid"]["MAPE_pct"], "비고": "고정 검증 11개 셀"},
        {"구분": "Test (Batch2)", "MAPE (%)": evaluations["test"]["MAPE_pct"], "비고": "고정 모델, 수명 확인 39개 셀"},
        *[{"구분": f"Gap ({name})", "MAPE (%)": value,
           "비고": {"Train-Valid": "Valid − Train CV (%p)", "Valid-Test": "Test − Valid (%p)",
                    "Target-Test": "Test − 9.1 (%p)"}[name]} for name, value in gaps.items()],
    ])
    test_predictions = predictions.loc[predictions["split"].eq("test")].copy()
    test_predictions["life_group"] = np.where(test_predictions["cycle_life"].lt(500), "short (<500)", "other (>=500)")
    diagnostics = [{"group": name, "n": len(group), **metrics(group["cycle_life"], group["predicted_cycle_life"])}
                   for name, group in test_predictions.groupby("life_group")]
    model_inputs = pd.concat([state["batch1"], test], ignore_index=True)[KEYS + columns + ["cycle_life"]]
    model_inputs.to_csv(DATA / "battery_features.csv", index=False)
    assert len(model_inputs) == 85 and "batch3" not in set(model_inputs["batch"])
    performance.to_csv(out / "model_performance.csv", index=False)
    predictions.to_csv(out / "model_predictions.csv", index=False)
    coefficients = []
    regressor, scaler = model.named_steps["regressor"], model.named_steps["scale"]
    if hasattr(regressor, "coef_"):
        raw = regressor.coef_ / scaler.scale_
        intercept = float(regressor.intercept_ - raw @ scaler.mean_)
        coefficients = [{"feature": f, "standardized_coef": float(s), "raw_coef": float(r)}
                        for f, s, r in zip(columns, regressor.coef_, raw)]
        assert np.allclose(intercept + test[columns].to_numpy() @ raw, test_prediction)
    else:
        intercept = None
    manifest = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "model": selected["candidate"], "features": columns, "target": "cycle_life (raw cycles)",
        "prediction_cycle": 100, "seed": SEED, "target_MAPE_pct": TARGET_MAPE,
        "selection_batch": "batch1", "test_used_for_selection": False,
        "train_batch": "batch1", "test_batch": "batch2", "excluded_batches": ["batch3"],
        "train_cv_MAPE_pct": train_mape, "train_cv_std_pp": selected["CV_std_pp"],
        "fold_metrics": folds, "evaluations": evaluations, "gaps_pp": gaps,
        "selected_params": selected["params"], "selection_rule": selection["selection_rule"],
        "minimum_CV_candidate": selection["minimum_CV_candidate"],
        "one_se_threshold_MAPE_pct": selection["one_se_threshold_MAPE_pct"],
        "training_rows": selection["training_rows"], "validation_rows": selection["validation_rows"],
        "missing_test_target_rows": missing_labels, "test_diagnostics": diagnostics,
        "coefficients": coefficients, "raw_intercept": intercept,
        "python": selection["python"], "platform": selection["platform"], "packages": selection["packages"],
        "requirements_sha256": selection["requirements_sha256"],
        "input_file_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                              for name in ["batch1_model_candidates.csv", "battery_features.csv", "batch2_cells.csv"]},
        "evaluation_scope": "Assignment cross-batch evaluation; Batch2 had prior exploratory exposure, not a prospectively blinded benchmark.",
    }
    (out / "model_metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, title in zip(axes, ["valid", "test"], ["Batch1 hold-out", "Batch2 test"]):
        frame = predictions.loc[predictions["split"].eq(name)]
        ax.scatter(frame["cycle_life"], frame["predicted_cycle_life"], s=38, alpha=0.85)
        bounds = [min(frame["cycle_life"].min(), frame["predicted_cycle_life"].min()) - 40,
                  max(frame["cycle_life"].max(), frame["predicted_cycle_life"].max()) + 40]
        ax.plot(bounds, bounds, "--", color="gray", linewidth=1)
        ax.set(xlim=bounds, ylim=bounds, xlabel="Actual cycle life", ylabel="Predicted cycle life",
               title=f"{title}: MAPE={evaluations[name]['MAPE_pct']:.2f}%")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "model_predictions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert selection_bytes == (out / "model_selection_manifest.json").read_bytes()
    restored = joblib.load(out / "battery_model.joblib")
    assert np.allclose(restored.predict(test[columns]), test_prediction)
    pd.testing.assert_frame_equal(pd.read_csv(out / "model_performance.csv"), performance)
    pd.testing.assert_frame_equal(pd.read_csv(out / "model_predictions.csv"), predictions, check_dtype=False)
    print(performance.round(3))
    print("선정 고정·분할·저장 모델·예측 재현 검증 완료")
    return {"performance": performance, "predictions": predictions, "manifest": manifest}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch1 CV 모델 선정 및 Batch2 고정 모델 평가")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--selection-only", action="store_true", help="Batch1만으로 모델 선정; Batch2는 읽지 않음")
    args = parser.parse_args()
    state = select_model(args.output_dir)
    if not args.selection_only:
        evaluate_model(state)
