"""Inputs are pandas DataFrames whose numeric features have already been binned
by main.ID3Preprocessor. Labels are aligned Series: 0=Graduate, 1=Dropout.
Do not replace these TODOs with a library classifier.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TreeNode:
    """Keep training class counts even at internal nodes for unseen values."""

    counts: tuple[int, int]  # (Graduate count, Dropout count)
    prediction: int  # majority class; choose 0 on a tie
    feature: str | None = None  # None means leaf
    children: dict[Any, "TreeNode"] = field(default_factory=dict)


class ID3Classifier:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 10):
        """Root depth is 0. max_depth=0 produces a root leaf.

        min_samples_split limits the parent size, not each child's size.
        fit must reject negative depth or min_samples_split < 2.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: TreeNode | None = None
        self.feature_names_: list[str] = []

    @staticmethod
    def entropy(y: pd.Series | np.ndarray) -> float:
        """Binary Shannon entropy in bits; 0 for empty or pure labels.

        H(S) = -sum_c p_c * log2(p_c), omitting zero-probability terms.
        """
        values = np.asarray(y)
        if values.size == 0:
            return 0.0
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def information_gain(values: pd.Series, y: pd.Series) -> float:
        """H(S) - weighted-sum H(S_v) over the values of one feature."""
        parent_entropy = ID3Classifier.entropy(y)
        n_total = len(y)
        weighted_child_entropy = 0.0
        for value in values.unique():
            mask = values.eq(value)
            branch = y.loc[mask]
            weighted_child_entropy += (len(branch) / n_total) * ID3Classifier.entropy(branch)
        return parent_entropy - weighted_child_entropy

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ID3Classifier":
        """Validate inputs, store schema, build the tree, and return self."""
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be None or a nonnegative integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if X.empty:
            raise ValueError("Training data must be nonempty")
        if X.isna().any().any():
            raise ValueError("Training features must not contain missing values")
        if len(y) != len(X):
            raise ValueError("X and y must have the same length")
        if not y.index.equals(X.index):
            raise ValueError("X and y indexes must be aligned")
        if not X.index.is_unique or not y.index.is_unique:
            raise ValueError("Row indexes must be unique")
        labels = pd.unique(y.to_numpy())
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("Labels must be binary 0/1")
        self.feature_names_ = list(X.columns)
        self.root_ = self._build_tree(X, y, self.feature_names_, depth=0)
        return self

    def _build_tree(
        self, X: pd.DataFrame, y: pd.Series, features: list[str], depth: int
    ) -> TreeNode:
        """Recursively build a categorical ID3 subtree."""
        n_graduate = int((y == 0).sum())
        n_dropout = int((y == 1).sum())
        node = TreeNode(
            counts=(n_graduate, n_dropout),
            prediction=int(n_dropout > n_graduate),
        )

        # Stopping: pure node, exhausted features, depth limit, or too few rows.
        if n_dropout == 0 or n_graduate == 0:
            return node
        if not features:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node

        best_feature = None
        best_gain = 1e-12
        for feature in features:  # Original column order resolves equal-gain ties.
            gain = self.information_gain(X[feature], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        if best_feature is None:
            return node

        node.feature = best_feature
        remaining = [feature for feature in features if feature != best_feature]
        for value in X[best_feature].unique():
            mask = X[best_feature].eq(value)
            child = self._build_tree(X.loc[mask], y.loc[mask], remaining, depth + 1)
            node.children[value] = child
        return node

    def _traverse(self, row: pd.Series) -> TreeNode:
        """Return the leaf or the deepest node reached on an unseen branch."""
        node = self.root_
        while node.feature is not None:
            value = row[node.feature]
            if value not in node.children:
                return node
            node = node.children[value]
        return node

    def _check_predict_schema(self, X: pd.DataFrame) -> None:
        if self.root_ is None:
            raise ValueError("Fit the tree before predicting")
        if list(X.columns) != self.feature_names_:
            raise ValueError("Query columns must match the fitted feature schema")

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Empirical dropout fraction (counts[1]/total) at the reached node."""
        self._check_predict_schema(X)
        risks = np.empty(len(X), dtype=float)
        for position, (_, row) in enumerate(X.iterrows()):
            node = self._traverse(row)
            total = node.counts[0] + node.counts[1]
            risks[position] = node.counts[1] / total if total > 0 else 0.0
        return risks

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Majority class at the reached node; an exact tie predicts 0."""
        self._check_predict_schema(X)
        predictions = np.empty(len(X), dtype=int)
        for position, (_, row) in enumerate(X.iterrows()):
            predictions[position] = self._traverse(row).prediction
        return predictions

    def decision_path(self, row: pd.Series) -> list[dict[str, Any]]:
        """Explain a single transformed row, node by node."""
        if self.root_ is None:
            raise ValueError("Fit the tree before requesting a path")
        if list(row.index) != self.feature_names_:
            raise ValueError("Row index must match the fitted feature schema")
        path: list[dict[str, Any]] = []
        node = self.root_
        while True:
            if node.feature is None:
                path.append({
                    "feature": None, "value": None,
                    "counts": node.counts, "fallback": False,
                })
                return path
            value = row[node.feature]
            if value not in node.children:
                path.append({
                    "feature": node.feature, "value": value,
                    "counts": node.counts, "fallback": True,
                })
                return path
            path.append({
                "feature": node.feature, "value": value,
                "counts": node.counts, "fallback": False,
            })
            node = node.children[value]

    def tree_stats(self) -> dict[str, int]:
        """Return depth (root=0), node count, and leaf count."""
        if self.root_ is None:
            raise ValueError("Fit the tree before requesting statistics")

        def walk(node: TreeNode, depth: int) -> tuple[int, int, int]:
            """Return (deepest depth, number of nodes, number of leaves)."""
            if not node.children:
                return depth, 1, 1
            deepest = depth
            n_nodes = 1
            n_leaves = 0
            for child in node.children.values():
                child_depth, child_nodes, child_leaves = walk(child, depth + 1)
                deepest = max(deepest, child_depth)
                n_nodes += child_nodes
                n_leaves += child_leaves
            return deepest, n_nodes, n_leaves

        depth, n_nodes, n_leaves = walk(self.root_, depth=0)
        return {"depth": depth, "n_nodes": n_nodes, "n_leaves": n_leaves}
