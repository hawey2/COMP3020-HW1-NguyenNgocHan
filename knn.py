"""For distance='mixed', main.KNNPreprocessor scales numeric columns and leaves
nominal codes unchanged. For distance='raw', pass unscaled original features.
DataFrame indexes are unique integer training row IDs, never distance features.
"""

import numpy as np
import pandas as pd


class KNNClassifier:
    def __init__(
        self,
        k: int = 5,
        distance: str = "mixed",
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ):
        self.k = k
        self.distance = distance
        self.numeric_columns = list(numeric_columns or [])
        self.categorical_columns = list(categorical_columns or [])
        self.X_train_: pd.DataFrame | None = None
        self.y_train_: pd.Series | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KNNClassifier":
        """Validate inputs, store copies of the training rows, and return self."""
        if isinstance(self.k, bool) or not isinstance(self.k, (int, np.integer)):
            raise ValueError("k must be an integer")
        if self.k < 1 or self.k > len(X):
            raise ValueError("k must lie within [1, number of training rows]")
        if self.distance not in ("raw", "mixed"):
            raise ValueError("distance must be 'raw' or 'mixed'")
        if X.empty:
            raise ValueError("Training data must be nonempty")
        if X.isna().any().any():
            raise ValueError("Training features must not contain missing values")
        if len(y) != len(X) or not y.index.equals(X.index):
            raise ValueError("X and y indexes must be aligned")
        if not X.index.is_unique or not y.index.is_unique:
            raise ValueError("Row indexes must be unique")
        labels = pd.unique(y.to_numpy())
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("Labels must be binary 0/1")
        if self.distance == "mixed":
            numeric_set, categorical_set = set(self.numeric_columns), set(self.categorical_columns)
            if numeric_set & categorical_set:
                raise ValueError("numeric and categorical columns must be disjoint")
            if (numeric_set | categorical_set) != set(X.columns):
                raise ValueError("numeric and categorical columns must cover X exactly")
        self.feature_names_ = list(X.columns)
        self.X_train_ = X.copy(deep=True)
        self.y_train_ = y.copy(deep=True)
        return self

    def _distances(self, row: pd.Series) -> np.ndarray:
        """Distance from one query Series to every stored training row."""
        if self.X_train_ is None:
            raise ValueError("Fit kNN before calculating distances")
        squared = np.zeros(len(self.X_train_), dtype=float)
        if self.distance == "raw":
            for column in self.feature_names_:
                difference = float(row[column]) - self.X_train_[column].to_numpy(dtype=float)
                squared += difference ** 2
        else:
            for column in self.numeric_columns:
                difference = float(row[column]) - self.X_train_[column].to_numpy(dtype=float)
                squared += difference ** 2
            for column in self.categorical_columns:
                mismatch = (row[column] != self.X_train_[column].to_numpy()).astype(float)
                squared += mismatch
        return np.sqrt(squared)

    def kneighbors(
        self, X: pd.DataFrame, n_neighbors: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, train_row_ids) for the nearest neighbors."""
        if self.X_train_ is None:
            raise ValueError("Fit kNN before querying neighbors")
        if list(X.columns) != self.feature_names_:
            raise ValueError("Query columns must match the fitted feature schema")
        if X.isna().any().any():
            raise ValueError("Query rows must not contain missing values")
        requested_k = self.k if n_neighbors is None else n_neighbors
        if isinstance(requested_k, bool) or not isinstance(requested_k, (int, np.integer)):
            raise ValueError("n_neighbors must be an integer")
        n_train = len(self.X_train_)
        if requested_k < 1 or requested_k > n_train:
            raise ValueError("n_neighbors must lie within [1, number of training rows]")
        train_ids = self.X_train_.index.to_numpy()
        distances = np.empty((len(X), requested_k), dtype=float)
        row_ids = np.empty((len(X), requested_k), dtype=int)
        for query_number, (_, row) in enumerate(X.iterrows()):
            all_distances = self._distances(row)
            # Primary key = ascending distance, secondary = ascending row ID.
            order = np.lexsort((train_ids, all_distances))
            distances[query_number] = all_distances[order][:requested_k]
            row_ids[query_number] = train_ids[order][:requested_k]
        return distances, row_ids

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Fraction of the k nearest training neighbors that are dropouts."""
        _, row_ids = self.kneighbors(X)
        risks = np.empty(len(X), dtype=float)
        for query_number in range(len(X)):
            neighbor_labels = self.y_train_.reindex(row_ids[query_number]).to_numpy()
            risks[query_number] = neighbor_labels.mean()
        return risks

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Majority vote over the k neighbors; an exact tie predicts 0."""
        return (self.predict_risk(X) > 0.5).astype(int)
