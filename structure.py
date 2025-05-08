import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample classification data (fixed parameters)
X, y = make_classification(
    n_samples=100, n_features=5, n_informative=3,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simple Clonal Selection Algorithm
def clonal_selection(X_train, y_train, pop_size=10, generations=5):
    population = np.random.randint(1, 10, size=(pop_size, 1))  # max_depth values
    best_depth = 1
    best_score = 0

    for gen in range(generations):
        for i in range(pop_size):
            depth = int(population[i][0])
            model = DecisionTreeClassifier(max_depth=depth)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_train, model.predict(X_train))

            # Save best depth
            if acc > best_score:
                best_score = acc
                best_depth = depth

        # Clone best and mutate one random individual
        mutation = best_depth + np.random.randint(-1, 2)
        mutation = max(1, min(mutation, 10))  # keep in range 1-10
        population[np.random.randint(0, pop_size)] = [mutation]

        print(f"Generation {gen+1}: Best max_depth = {best_depth}, Accuracy = {best_score:.2f}")

    return best_depth

# Run the algorithm
best_depth = clonal_selection(X_train, y_train)

# Final evaluation on test data
final_model = DecisionTreeClassifier(max_depth=best_depth)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print(f"\nTest Accuracy with best max_depth ({best_depth}): {accuracy_score(y_test, y_pred):.2f}")
