"""초기 100사이클의 세 지표로 수명을 예측하는 Ridge 학습·평가 스크립트.

Batch1 프로토콜 단위 Hold-out과 5-fold CV로 학습하고 Batch2를 평가한다.
실행: .venv/bin/python model/train_ridge.py
산출물 경로 변경: --output-dir /tmp/ridge-results
"""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def main(output_dir: Path = ROOT / "outputs") -> None:
    # 1. 데이터 로드와 프로토콜 확인
    DATA = ROOT / "data" / "processed"
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    SEED = 42
    TARGET_MAPE = 9.1
    FEATURES = ["log10_dqv_var", "charge_stress", "joule_heat_proxy"]
    KEYS = ["batch", "cell_id"]
    ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    feature_table = pd.read_csv(DATA / "battery_features.csv")
    data = feature_table.loc[feature_table["batch"].isin(["batch1", "batch2"])].copy()
    policies = pd.concat([
        pd.read_csv(DATA / f"{batch}_cells.csv").assign(batch=batch)
        for batch in ["batch1", "batch2"]
    ], ignore_index=True)
    assert not data.duplicated(KEYS).any() and not policies.duplicated(KEYS).any()
    data = data.merge(policies[KEYS + ["policy"]], on=KEYS, how="left", validate="one_to_one")
    # 세 숫자 조건으로 프로토콜을 묶어 표기나 접미사 차이로 그룹이 나뉘는 것을 막는다.
    numeric_policy = data["policy"].str.extract(r"^([\d.]+)C\((\d+)%\)-([\d.]+)C")
    assert numeric_policy.notna().all().all(), "충전 프로토콜 파싱 실패"
    data["protocol"] = numeric_policy.astype(float).astype(str).agg("|".join, axis=1)
    data = data.sort_values(KEYS).reset_index(drop=True)
    assert set(data["batch"]) == {"batch1", "batch2"}
    assert np.isfinite(data[FEATURES + ["cycle_life"]].to_numpy()).all()
    assert data["cycle_life"].gt(100).all()

    excluded = policies.loc[policies["cycle_life"].notna()].merge(
        data[KEYS], on=KEYS, how="left", indicator=True, validate="one_to_one")
    excluded = excluded.loc[excluded["_merge"].eq("left_only"), KEYS + ["policy", "cycle_life"]]
    print("모델링 대상:")
    print(data.groupby("batch").agg(n_cells=("cell_id", "size"), n_protocols=("protocol", "nunique")))
    print("Feature Engineering에서 IR=0으로 제외한 셀:")
    print(excluded.reset_index(drop=True))

    # 2. Batch1 Hold-out과 그룹 CV 분할
    batch1 = data.loc[data["batch"].eq("batch1")].reset_index(drop=True)
    test = data.loc[data["batch"].eq("batch2")].reset_index(drop=True)
    train_idx, valid_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    ).split(batch1, groups=batch1["protocol"]))
    train = batch1.iloc[train_idx].reset_index(drop=True)
    valid = batch1.iloc[valid_idx].reset_index(drop=True)
    assert set(train["cell_id"]).isdisjoint(valid["cell_id"])
    assert set(train["protocol"]).isdisjoint(valid["protocol"])
    assert len(train) + len(valid) == len(batch1)
    assert len(test) > 0 and train["protocol"].nunique() >= 5

    cv_splits = list(GroupKFold(n_splits=5).split(train, groups=train["protocol"]))
    coverage = np.zeros(len(train), dtype=int)
    for fit_idx, check_idx in cv_splits:
        assert set(train.iloc[fit_idx]["protocol"]).isdisjoint(train.iloc[check_idx]["protocol"])
        coverage[check_idx] += 1
    assert np.all(coverage == 1)
    split_summary = pd.DataFrame([
        {"구분": label, "셀 수": len(frame), "프로토콜 수": frame["protocol"].nunique(),
         "수명 최솟값": frame["cycle_life"].min(), "수명 최댓값": frame["cycle_life"].max()}
        for label, frame in [("Train (Batch1)", train), ("Valid (Batch1 Hold-out)", valid), ("Test (Batch2)", test)]
    ])
    print(split_summary)

    train_feature_corr = train[FEATURES].corr()
    print("Batch1 학습 데이터의 입력 간 Pearson 상관:")
    print(train_feature_corr.round(3))

    # 3. 입력 구성 비교와 Ridge 규제 강도 선택
    feature_sets = {
        "Ridge (L)": FEATURES[:1],
        "Ridge (L+F)": FEATURES[:2],
        "Ridge (L+F+H)": FEATURES,
    }
    models, searches, comparison_rows = {}, {}, []
    for name, columns in feature_sets.items():
        pipeline = Pipeline([("scale", StandardScaler()), ("regressor", Ridge())])
        search = GridSearchCV(
            pipeline, {"regressor__alpha": ALPHAS}, cv=cv_splits,
            scoring="neg_mean_absolute_percentage_error", n_jobs=1, error_score="raise"
        )
        search.fit(train[columns], train["cycle_life"])
        models[name], searches[name] = search.best_estimator_, search
        comparison_rows.append({
            "model": name, "alpha": float(search.best_params_["regressor__alpha"]),
            "Train_CV_MAPE_pct": float(-100 * search.best_score_),
            "CV_std_pp": float(100 * search.cv_results_["std_test_score"][search.best_index_]),
            "Valid_MAPE_pct": float(100 * mean_absolute_percentage_error(
                valid["cycle_life"], search.predict(valid[columns])))
        })

    baseline = Pipeline([("scale", StandardScaler()), ("regressor", LinearRegression())])
    baseline_scores = -100 * cross_val_score(
        baseline, train[FEATURES], train["cycle_life"], cv=cv_splits,
        scoring="neg_mean_absolute_percentage_error", error_score="raise"
    )
    baseline.fit(train[FEATURES], train["cycle_life"])
    comparison_rows.append({
        "model": "LinearRegression (L+F+H)", "alpha": None,
        "Train_CV_MAPE_pct": float(baseline_scores.mean()), "CV_std_pp": float(baseline_scores.std()),
        "Valid_MAPE_pct": float(100 * mean_absolute_percentage_error(
            valid["cycle_life"], baseline.predict(valid[FEATURES])))
    })
    comparison = pd.DataFrame(comparison_rows)
    print(comparison.round(3))

    final_name = "Ridge (L+F+H)"
    final_model, final_search = models[final_name], searches[final_name]
    best_alpha = float(final_search.best_params_["regressor__alpha"])
    scaler = final_model.named_steps["scale"]
    assert scaler.n_samples_seen_ == len(train)
    assert np.allclose(scaler.mean_, train[FEATURES].mean().to_numpy())
    alpha_results = pd.DataFrame({
        "alpha": [float(params["regressor__alpha"]) for params in final_search.cv_results_["params"]],
        "CV_MAPE_pct": -100 * final_search.cv_results_["mean_test_score"],
        "CV_std_pp": 100 * final_search.cv_results_["std_test_score"]
    })
    print(f"최종 모델: {final_name}, alpha={best_alpha:g}")
    print(alpha_results.round(3))

    # 4. 고정 모델 평가와 Gap 계산
    def regression_metrics(y, prediction):
        y, prediction = np.asarray(y), np.asarray(prediction)
        assert len(y) == len(prediction) and np.isfinite(prediction).all() and (y > 0).all()
        return {
            "MAPE_pct": float(100 * mean_absolute_percentage_error(y, prediction)),
            "MAE_cycles": float(mean_absolute_error(y, prediction)),
            "RMSE_cycles": float(root_mean_squared_error(y, prediction)),
            "bias_cycles": float(np.mean(prediction - y)),
        }

    prediction_frames, fold_rows = [], []
    # ponytail: 선택한 α의 CV 점수를 보고한다. 별도 검증은 고정 Hold-out으로 확인한다.
    for fold, (fit_idx, check_idx) in enumerate(cv_splits, 1):
        fit_data, check_data = train.iloc[fit_idx], train.iloc[check_idx]
        fold_model = clone(final_model).fit(fit_data[FEATURES], fit_data["cycle_life"])
        prediction = fold_model.predict(check_data[FEATURES])
        fold_rows.append({"fold": fold, "n": len(check_data),
                          **regression_metrics(check_data["cycle_life"], prediction)})
        prediction_frames.append(check_data.assign(split="train_cv", fold=fold, predicted_cycle_life=prediction))
    fold_metrics = pd.DataFrame(fold_rows)
    train_mape = float(fold_metrics["MAPE_pct"].mean())
    assert np.isclose(train_mape, -100 * final_search.best_score_)
    assert np.allclose(fold_metrics["MAPE_pct"], [
        -100 * final_search.cv_results_[f"split{i}_test_score"][final_search.best_index_] for i in range(5)
    ])

    evaluations = {}
    for name, frame in [("valid", valid), ("test", test)]:
        prediction = final_model.predict(frame[FEATURES])
        evaluations[name] = {"n": len(frame), **regression_metrics(frame["cycle_life"], prediction)}
        prediction_frames.append(frame.assign(split=name, fold=0, predicted_cycle_life=prediction))
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["split", *KEYS]).reset_index(drop=True)
    predictions["error_cycles"] = predictions["predicted_cycle_life"] - predictions["cycle_life"]
    predictions["APE_pct"] = 100 * predictions["error_cycles"].abs() / predictions["cycle_life"]
    assert not predictions.duplicated(KEYS).any() and len(predictions) == len(data)
    assert set(predictions["batch"]) == {"batch1", "batch2"}

    valid_mape, test_mape = evaluations["valid"]["MAPE_pct"], evaluations["test"]["MAPE_pct"]
    gaps = {"Train-Valid": valid_mape - train_mape, "Valid-Test": test_mape - valid_mape,
            "Target-Test": test_mape - TARGET_MAPE}
    performance = pd.DataFrame([
        {"구분": "Train (Batch1 CV)", "MAPE (%)": train_mape, "비고": f"학습 {len(train)}개 셀, 프로토콜 5-fold 평균"},
        {"구분": "Valid (Batch1 Hold-out)", "MAPE (%)": valid_mape, "비고": f"고정 검증 {len(valid)}개 셀"},
        {"구분": "Test (Batch2)", "MAPE (%)": test_mape, "비고": f"배치 간 평가 {len(test)}개 셀"},
        {"구분": "Gap (Train-Valid)", "MAPE (%)": gaps["Train-Valid"], "비고": "Valid − Train CV (%p)"},
        {"구분": "Gap (Valid-Test)", "MAPE (%)": gaps["Valid-Test"], "비고": "Test − Valid (%p)"},
        {"구분": "Gap (Target-Test)", "MAPE (%)": gaps["Target-Test"], "비고": "Test − 9.1 (%p)"},
    ])
    print(performance.round(3))
    print(fold_metrics.round(3))
    print(pd.DataFrame(evaluations).T.round(3))

    # 5. 배치별 오차 해석과 예측 그래프
    test_predictions = predictions.loc[predictions["split"].eq("test")].copy()
    test_predictions["variant"] = np.where(
        test_predictions["policy"].str.contains("newstructure", case=False), "Newstructure", "Base")
    test_predictions["life_group"] = np.where(test_predictions["cycle_life"].lt(500), "short (<500)", "other (>=500)")
    diagnostic_rows = []
    for dimension in ["variant", "life_group"]:
        for name, group in test_predictions.groupby(dimension):
            diagnostic_rows.append({"dimension": dimension, "group": name, "n": len(group),
                **regression_metrics(group["cycle_life"], group["predicted_cycle_life"])})
    diagnostics = pd.DataFrame(diagnostic_rows)
    print(diagnostics.round(3))

    regressor = final_model.named_steps["regressor"]
    raw_coefficients = regressor.coef_ / scaler.scale_
    raw_intercept = float(regressor.intercept_ - raw_coefficients @ scaler.mean_)
    coefficients = pd.DataFrame({"feature": FEATURES, "standardized_coef": regressor.coef_,
                                 "raw_coef": raw_coefficients})
    assert np.allclose(raw_intercept + test[FEATURES].to_numpy() @ raw_coefficients,
                       final_model.predict(test[FEATURES]))
    print(coefficients.round(3))
    print(f"원래 변수 단위의 절편: {raw_intercept:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, title in zip(axes, ["valid", "test"], ["Batch1 hold-out", "Batch2 test"]):
        frame = predictions.loc[predictions["split"].eq(name)]
        ax.scatter(frame["cycle_life"], frame["predicted_cycle_life"], s=38, alpha=0.85)
        limits = [min(frame["cycle_life"].min(), frame["predicted_cycle_life"].min()) - 40,
                  max(frame["cycle_life"].max(), frame["predicted_cycle_life"].max()) + 40]
        ax.plot(limits, limits, "--", color="gray", linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel="Actual cycle life", ylabel="Predicted cycle life",
               title=f"{title}: MAPE={evaluations[name]['MAPE_pct']:.2f}%")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "model_predictions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


    # 6. 모델·평가 결과 저장 및 재현 검사
    performance.to_csv(OUT / "model_performance.csv", index=False)
    predictions.to_csv(OUT / "model_predictions.csv", index=False)
    joblib.dump(final_model, OUT / "ridge_model.joblib")
    manifest = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "model": final_name, "features": FEATURES, "target": "cycle_life (raw cycles)",
        "prediction_cycle": 100, "alpha": best_alpha, "alpha_candidates": ALPHAS,
        "seed": SEED, "target_MAPE_pct": TARGET_MAPE,
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "packages": {package: version(package) for package in
                     ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "joblib"]},
        "requirements_sha256": hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest(),
        "input_file_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                              for name in ["battery_features.csv", "batch1_cells.csv", "batch2_cells.csv"]},
        "data_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "train_batch": "batch1", "test_batch": "batch2", "excluded_batches": ["batch3"],
        "cv": "5-fold GroupKFold by numeric charging protocol; selected-alpha CV mean",
        "holdout": "GroupShuffleSplit, test_size=0.2 of protocols, random_state=42",
        "training_rows": train[KEYS + ["policy", "protocol"]].to_dict("records"),
        "validation_rows": valid[KEYS + ["policy", "protocol"]].to_dict("records"),
        "excluded_feature_rows": excluded.to_dict("records"),
        "train_cv_MAPE_pct": train_mape, "train_cv_std_pp": float(fold_metrics["MAPE_pct"].std(ddof=0)),
        "fold_metrics": fold_metrics.to_dict("records"), "evaluations": evaluations,
        "gaps_pp": gaps, "alpha_search": alpha_results.to_dict("records"),
        "comparison": [dict(row) for row in comparison_rows],
        "train_feature_correlations": train_feature_corr.to_dict(),
        "coefficients": coefficients.to_dict("records"), "raw_intercept": raw_intercept,
        "test_diagnostics": diagnostics.to_dict("records"),
        "interpretation_scope": "Batch2 was used in earlier EDA/feature review; assignment cross-batch evaluation."
    }
    (OUT / "model_metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    # 저장한 모델과 표가 실행 결과와 일치하는지 확인한다.
    restored_model = joblib.load(OUT / "ridge_model.joblib")
    assert np.allclose(restored_model.predict(test[FEATURES]), final_model.predict(test[FEATURES]))
    pd.testing.assert_frame_equal(pd.read_csv(OUT / "model_performance.csv"), performance)
    pd.testing.assert_frame_equal(pd.read_csv(OUT / "model_predictions.csv"), predictions, check_dtype=False)
    saved = json.loads((OUT / "model_metrics.json").read_text())
    assert saved["features"] == FEATURES and saved["test_batch"] == "batch2"
    assert np.isclose(saved["gaps_pp"]["Target-Test"], test_mape - 9.1)
    print("저장 및 분할·스케일링·예측 재현 검증 완료")
    print("성능표 / 셀별 예측 / 모델·설정 / 예측 그래프:", OUT)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch1 학습·검증과 Batch2 Ridge 평가")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="모델·평가 결과 저장 경로")
    main(parser.parse_args().output_dir)
