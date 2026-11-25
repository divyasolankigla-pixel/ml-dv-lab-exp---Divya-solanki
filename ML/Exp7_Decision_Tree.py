"""Experiment 7: Decision Tree Classifier on the scikit-learn Iris dataset."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )

    clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Experiment 7: Decision Tree Classifier (Iris) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nFeature Importances:")
    for name, importance in zip(iris.feature_names, clf.feature_importances_):
        print(f"  {name:<18}: {importance:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("Decision Tree Rules:")
    print(export_text(clf, feature_names=list(iris.feature_names)))


if __name__ == "__main__":
    main()
