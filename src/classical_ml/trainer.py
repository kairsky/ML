"""Classical ML training with GridSearchCV and pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class ClassicalMLTrainer:
    """Train and tune multiple sklearn classifiers with unified API."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        use_class_weight: bool = False,
    ) -> None:
        self.preprocessor = preprocessor
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_class_weight = use_class_weight
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.fitted_models: Dict[str, Pipeline] = {}
        self.search_results: Dict[str, Any] = {}

    def _build_estimators(self) -> Dict[str, Tuple[BaseEstimator, Dict[str, List[Any]]]]:
        cw = "balanced" if self.use_class_weight else None
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight=cw),
                {
                    "clf__C": [0.01, 0.1, 1.0, 10.0],
                    "clf__penalty": ["l2"],
                    "clf__solver": ["lbfgs", "saga"],
                },
            ),
            "random_forest": (
                RandomForestClassifier(random_state=self.random_state, class_weight=cw, n_jobs=self.n_jobs),
                {
                    "clf__n_estimators": [100, 200],
                    "clf__max_depth": [10, 20, None],
                    "clf__min_samples_split": [2, 5],
                },
            ),
            "svm": (
                SVC(probability=True, random_state=self.random_state, class_weight=cw),
                {
                    "clf__C": [0.1, 1.0, 10.0],
                    "clf__kernel": ["rbf", "linear"],
                    "clf__gamma": ["scale", "auto"],
                },
            ),
            "knn": (
                KNeighborsClassifier(n_jobs=self.n_jobs),
                {
                    "clf__n_neighbors": [3, 5, 7, 11],
                    "clf__weights": ["uniform", "distance"],
                    "clf__metric": ["euclidean", "minkowski"],
                },
            ),
        }

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grids: Optional[Dict[str, Dict]] = None,
        use_randomized: Optional[Dict[str, bool]] = None,
        n_iter: int = 20,
    ) -> Dict[str, Pipeline]:
        """Train all classifiers with grid or randomized search."""
        estimators = self._build_estimators()
        if param_grids:
            for name, grid in param_grids.items():
                if name in estimators:
                    estimators[name] = (estimators[name][0], grid)

        use_randomized = use_randomized or {}

        for name, (estimator, grid) in estimators.items():
            pipe = Pipeline([("prep", self.preprocessor), ("clf", estimator)])
            if use_randomized.get(name, False):
                search = RandomizedSearchCV(
                    pipe,
                    grid,
                    n_iter=n_iter,
                    cv=self.cv,
                    scoring="f1_weighted",
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    refit=True,
                    verbose=1,
                )
            else:
                search = GridSearchCV(
                    pipe,
                    grid,
                    cv=self.cv,
                    scoring="f1_weighted",
                    n_jobs=self.n_jobs,
                    refit=True,
                    verbose=1,
                )
            search.fit(X_train, y_train)
            self.fitted_models[name] = search.best_estimator_
            self.search_results[name] = {
                "best_params": search.best_params_,
                "best_score": float(search.best_score_),
                "cv_results": search.cv_results_,
            }
        return self.fitted_models

    def save_models(self, output_dir: str | Path) -> None:
        """Persist fitted pipelines."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.fitted_models.items():
            joblib.dump(model, output_dir / f"{name}.joblib")
