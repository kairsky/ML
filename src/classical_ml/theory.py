"""Theoretical documentation for classical ML algorithms."""

from __future__ import annotations

from typing import Any, Dict, List


ALGORITHM_THEORY: Dict[str, Dict[str, Any]] = {
    "k-NN": {
        "intuition": (
            "Instance-based learning: classify a point by majority vote of its k nearest "
            "neighbors in feature space using a distance metric d(x, x')."
        ),
        "decision_rule": "ŷ = mode({y_i : x_i ∈ N_k(x)}) where N_k(x) are k nearest neighbors.",
        "hyperparameters": ["n_neighbors (k)", "weights", "metric", "p (Minkowski)"],
        "training_complexity": "O(1) — lazy learner, stores training data.",
        "inference_complexity": "O(n·d) per query for n training samples, d features.",
        "advantages": ["Simple", "No training phase", "Naturally handles multiclass"],
        "limitations": ["Slow at inference", "Sensitive to scale", "Curse of dimensionality"],
        "best_use_cases": ["Small datasets", "Low-dimensional embeddings", "Baseline models"],
    },
    "Decision Tree": {
        "intuition": (
            "Recursively partition feature space to minimize impurity "
            "(Gini or entropy) at each node."
        ),
        "decision_rule": "Follow branches x_j ≤ t until leaf; predict leaf majority class.",
        "hyperparameters": ["max_depth", "min_samples_split", "min_samples_leaf", "criterion"],
        "training_complexity": "O(n·d·log n) per tree (greedy splits).",
        "inference_complexity": "O(depth) ≈ O(log n) per sample.",
        "advantages": ["Interpretable", "Handles mixed types", "Non-linear boundaries"],
        "limitations": ["High variance", "Overfitting", "Unstable to small data changes"],
        "best_use_cases": ["Tabular data with interactions", "Explainability requirements"],
    },
    "Random Forest": {
        "intuition": (
            "Bagging ensemble of decorrelated trees via bootstrap samples "
            "and random feature subsets; aggregate by majority vote."
        ),
        "decision_rule": "ŷ = mode({h_t(x)} for t=1..T trees).",
        "hyperparameters": ["n_estimators", "max_depth", "max_features", "min_samples_split"],
        "training_complexity": "O(T·n·d·log n) for T trees.",
        "inference_complexity": "O(T·depth) per sample.",
        "advantages": ["Robust", "Feature importance", "Low overfitting vs single tree"],
        "limitations": ["Less interpretable", "Large model size", "Slower than linear models"],
        "best_use_cases": ["General tabular classification", "Feature importance analysis"],
    },
    "SVM": {
        "intuition": (
            "Find maximum-margin hyperplane w·x + b = 0; kernel trick maps "
            "to higher dimensions for non-linear separation."
        ),
        "decision_rule": "ŷ = sign(Σ α_i y_i K(x_i, x) + b) for support vectors.",
        "hyperparameters": ["C", "kernel", "gamma", "degree"],
        "training_complexity": "O(n²) to O(n³) depending on solver.",
        "inference_complexity": "O(n_sv·d) for number of support vectors n_sv.",
        "advantages": ["Strong with high-dim data", "Kernel flexibility", "Margin maximization"],
        "limitations": ["Slow on large n", "Sensitive to scaling", "Probabilistic outputs need calibration"],
        "best_use_cases": ["Medium-sized datasets", "Text (linear kernel)", "Binary/multiclass (OvR)"],
    },
    "Logistic Regression": {
        "intuition": (
            "Model P(y=1|x) = σ(w·x + b) with sigmoid σ; optimize log-likelihood "
            "(cross-entropy) with L2 regularization."
        ),
        "decision_rule": "ŷ = 1 if P(y=1|x) ≥ 0.5 else 0 (multiclass: argmax softmax).",
        "hyperparameters": ["C (inverse regularization)", "penalty", "solver", "class_weight"],
        "training_complexity": "O(n·d·iterations) — convex optimization.",
        "inference_complexity": "O(d) per sample.",
        "advantages": ["Fast", "Probabilistic outputs", "Interpretable coefficients"],
        "limitations": ["Linear decision boundary (without features)", "Feature engineering needed"],
        "best_use_cases": ["Baseline linear classifier", "High-dimensional sparse text"],
    },
    "Naive Bayes": {
        "intuition": (
            "Apply Bayes' theorem P(y|x) ∝ P(y)∏ P(x_j|y) assuming feature independence."
        ),
        "decision_rule": "ŷ = argmax_y P(y) ∏_j P(x_j | y).",
        "hyperparameters": ["var_smoothing (Gaussian)", "alpha (Multinomial)", "binarize"],
        "training_complexity": "O(n·d) — count/means per class.",
        "inference_complexity": "O(c·d) for c classes.",
        "advantages": ["Very fast", "Works well with text", "Small data friendly"],
        "limitations": ["Independence assumption rarely holds", "Poor with correlated features"],
        "best_use_cases": ["Text classification", "Spam detection", "Quick baselines"],
    },
}


def get_comparison_table() -> List[Dict[str, str]]:
    """Return rows for algorithm comparison table."""
    rows = []
    for name, info in ALGORITHM_THEORY.items():
        rows.append(
            {
                "Algorithm": name,
                "Training": info["training_complexity"],
                "Inference": info["inference_complexity"],
                "Key hyperparameters": ", ".join(info["hyperparameters"][:3]),
                "Best use case": info["best_use_cases"][0],
            }
        )
    return rows
