import json
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "heart_failure_prediction_preprocessing"
TRAIN_PATH = DATA_DIR / "train_preprocessed.csv"
TEST_PATH = DATA_DIR / "test_preprocessed.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
TARGET_COLUMN = "HeartDisease"
RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def save_artifacts(model: RandomForestClassifier, X_train: pd.DataFrame, y_test: pd.Series, y_pred, y_proba) -> list[Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    paths = []

    confusion_matrix_path = ARTIFACT_DIR / "confusion_matrix.png"
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No Disease", "Disease"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    paths.append(confusion_matrix_path)

    roc_curve_path = ARTIFACT_DIR / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_curve_path, dpi=150)
    plt.close()
    paths.append(roc_curve_path)

    report_path = ARTIFACT_DIR / "classification_report.txt"
    report_path.write_text(classification_report(y_test, y_pred), encoding="utf-8")
    paths.append(report_path)

    feature_importance_path = ARTIFACT_DIR / "feature_importance.csv"
    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    paths.append(feature_importance_path)

    model_path = ARTIFACT_DIR / "random_forest_model.joblib"
    joblib.dump(model, model_path)
    paths.append(model_path)

    return paths


def main() -> None:
    X_train, X_test, y_train, y_test = load_dataset()

    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    artifact_paths = save_artifacts(model, X_train, y_test, y_pred, y_proba)
    params_path = ARTIFACT_DIR / "model_params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    artifact_paths.append(params_path)

    active_run = mlflow.active_run()
    run_context = nullcontext(active_run) if active_run else mlflow.start_run(run_name="heart_disease_ci_random_forest")

    with run_context as run:
        mlflow.set_tag("mlflow.runName", "heart_disease_ci_random_forest")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"RUN_ID={run.info.run_id}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
