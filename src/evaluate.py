from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from train import train_model

THRESHOLD = 0.35


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    print("\nMétricas:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(y_test, y_pred, target_names=["Permaneceu", "Cancelou"])
    )


if __name__ == "__main__":
    model, X_test, y_test = train_model()

    evaluate_model(model, X_test, y_test)
