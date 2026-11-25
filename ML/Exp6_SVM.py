"""Experiment 6: Support Vector Machine (SVM) Classification on Breast Cancer data."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
    )

    print("=== Experiment 6: Support Vector Machine (Breast Cancer) ===")
    best_model = None
    best_acc = -1.0

    for kernel in ("linear", "rbf", "poly"):
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, random_state=42))
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Kernel '{kernel:<6}' Test Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = (kernel, clf)

    best_kernel, best_clf = best_model
    y_pred = best_clf.predict(X_test)
    print(f"\nBest Kernel: {best_kernel} (Accuracy: {best_acc:.4f})")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))


if __name__ == "__main__":
    main()
