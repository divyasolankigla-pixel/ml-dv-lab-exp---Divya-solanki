"""Experiment 8: Gaussian Naive Bayes Classification on the Wine dataset."""

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main() -> None:
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.25, random_state=42, stratify=wine.target
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Experiment 8: Gaussian Naive Bayes (Wine) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nLearned Class Priors:")
    for name, prior in zip(wine.target_names, model.class_prior_):
        print(f"  {name:<10}: {prior:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))


if __name__ == "__main__":
    main()
