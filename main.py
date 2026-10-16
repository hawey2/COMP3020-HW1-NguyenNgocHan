"""Inspect data now: python main.py --inspect
After completing TODOs: python main.py --data dataset.csv --output results
Tests: python -m unittest discover -s tests -v
"""

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from decision_tree import ID3Classifier
from knn import KNNClassifier


SEED = 42
CHECKPOINTS = ("enrollment", "semester1", "semester2")
CATEGORICAL_COLUMNS = (
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
)
SEMESTER_FIELDS = ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")
SEMESTER1_COLUMNS = tuple(f"Curricular units 1st sem ({f})" for f in SEMESTER_FIELDS)
SEMESTER2_COLUMNS = tuple(f"Curricular units 2nd sem ({f})" for f in SEMESTER_FIELDS)
NUMERIC_COLUMNS = (
    "Application order", "Age at enrollment", *SEMESTER1_COLUMNS,
    *SEMESTER2_COLUMNS, "Unemployment rate", "Inflation rate", "GDP",
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Provided I/O helper: preserve original row positions as anonymous IDs."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.index = pd.Index(np.arange(len(df)), name="row_id")
    return df


def prepare_cohorts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (labeled_X, binary_y, enrolled_X) with original row indexes kept.

    Dropout -> 1, Graduate -> 0; the 794 Enrolled rows are set aside unchanged.
    """
    if "Target" not in df.columns:
        raise ValueError("DataFrame must contain a 'Target' column")
    target = df["Target"]
    if target.isna().any():
        raise ValueError("Target column contains missing labels")
    valid = {"Graduate", "Dropout", "Enrolled"}
    unknown = set(target.unique()) - valid
    if unknown:
        raise ValueError(f"Unknown Target labels: {sorted(unknown)}")
    labeled_mask = target.isin(("Graduate", "Dropout"))
    labeled = df.loc[labeled_mask]
    enrolled = df.loc[~labeled_mask]
    y = labeled["Target"].map({"Graduate": 0, "Dropout": 1})
    X = labeled.drop(columns=["Target"])
    enrolled = enrolled.drop(columns=["Target"])
    return X, y, enrolled


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    """Return nested checkpoint predictor lists in the original column order."""
    predictors = [c for c in columns if c not in ("Target", "row_id")]
    semester1 = set(SEMESTER1_COLUMNS)
    semester2 = set(SEMESTER2_COLUMNS)
    return {
        "enrollment": [c for c in predictors if c not in semester1 and c not in semester2],
        "semester1": [c for c in predictors if c not in semester2],
        "semester2": list(predictors),
    }


def split_row_ids(y: pd.Series, seed: int = SEED) -> dict[str, np.ndarray]:
    """Stratified 60/20/20 split of row IDs; reserve 20% test, then 25% of rest."""
    from sklearn.model_selection import train_test_split

    ids = np.asarray(y.index)
    labels = y.to_numpy()
    train_val_ids, test_ids = train_test_split(
        ids, test_size=0.2, random_state=seed, stratify=labels,
    )
    train_ids, validation_ids = train_test_split(
        train_val_ids, test_size=0.25, random_state=seed,
        stratify=y.loc[train_val_ids].to_numpy(),
    )
    return {"train": train_ids, "validation": validation_ids, "test": test_ids}


class ID3Preprocessor:
    def __init__(self, numeric_columns: list[str], n_bins: int = 3):
        self.numeric_columns = list(numeric_columns)
        self.n_bins = n_bins
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "ID3Preprocessor":
        """Learn internal quantile boundaries from training rows only."""
        if X.empty:
            raise ValueError("Training data must be nonempty")
        if X.isna().any().any():
            raise ValueError("Training features must not contain missing values")
        for column in self.numeric_columns:
            if column not in X.columns:
                raise ValueError(f"Numeric column '{column}' is missing from X")
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, (int, np.integer)) or self.n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        self.feature_names_ = list(X.columns)
        quantile_levels = np.arange(1, self.n_bins) / self.n_bins
        self.bin_edges_ = {}
        for column in self.numeric_columns:
            values = X[column].to_numpy(dtype=float)
            edges = np.quantile(values, quantile_levels, method="linear")
            edges = np.unique(edges)
            value_min, value_max = float(values.min()), float(values.max())
            edges = edges[(edges > value_min) & (edges < value_max)]
            self.bin_edges_[column] = edges
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bin numeric columns; boundaries from fit are reused unchanged."""
        if not self._fitted:
            raise ValueError("Fit the preprocessor before transforming")
        if list(X.columns) != self.feature_names_:
            raise ValueError("Query columns must match the fitted feature schema")
        if X.isna().any().any():
            raise ValueError("Query features must not contain missing values")
        transformed = X.copy(deep=True)
        for column, edges in self.bin_edges_.items():
            transformed[column] = np.searchsorted(edges, X[column].to_numpy(dtype=float), side="right")
        return transformed


class KNNPreprocessor:
    def __init__(self, numeric_columns: list[str]):
        self.numeric_columns = list(numeric_columns)
        self.means_: dict[str, float] = {}
        self.scales_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "KNNPreprocessor":
        """Learn training-only mean and population std (ddof=0)."""
        if X.empty:
            raise ValueError("Training data must be nonempty")
        if X.isna().any().any():
            raise ValueError("Training features must not contain missing values")
        for column in self.numeric_columns:
            if column not in X.columns:
                raise ValueError(f"Numeric column '{column}' is missing from X")
        self.feature_names_ = list(X.columns)
        self.means_ = {}
        self.scales_ = {}
        for column in self.numeric_columns:
            values = X[column].to_numpy(dtype=float)
            self.means_[column] = float(values.mean())
            scale = float(values.std(ddof=0))
            self.scales_[column] = scale if scale > 0 else 1.0
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace numeric columns with (x - mean)/scale; keep the rest."""
        if not self._fitted:
            raise ValueError("Fit the preprocessor before transforming")
        if list(X.columns) != self.feature_names_:
            raise ValueError("Query columns must match the fitted feature schema")
        if X.isna().any().any():
            raise ValueError("Query features must not contain missing values")
        transformed = X.copy(deep=True)
        for column in self.numeric_columns:
            transformed[column] = (X[column].to_numpy(dtype=float) - self.means_[column]) / self.scales_[column]
        return transformed


def select_advising_list(risks: pd.Series, capacity: float = 0.10) -> pd.Series:
    """Boolean mask: highest risks first, ties broken by ascending row ID."""
    if not (0.0 < capacity <= 1.0):
        raise ValueError("capacity must lie in (0, 1]")
    if len(risks) == 0:
        raise ValueError("risks must be nonempty")
    risk_values = risks.to_numpy(dtype=float)
    if not np.isfinite(risk_values).all():
        raise ValueError("risk scores must be finite")
    if (risk_values < 0).any() or (risk_values > 1).any():
        raise ValueError("risk scores must lie in [0, 1]")
    if not risks.index.is_unique:
        raise ValueError("risk score indexes must be unique")
    n = len(risks)
    budget = max(1, int(np.floor(capacity * n)))
    order = np.lexsort((risks.index.to_numpy(), -risk_values))
    chosen = risks.index[order][:budget]
    selected = pd.Series(False, index=risks.index, dtype=bool)
    selected.loc[chosen] = True
    return selected


def capacity_metrics(
    y: pd.Series, risks: pd.Series, capacity: float = 0.10
) -> dict[str, Any]:
    """Select a 10% advising list and return the capacity metrics described in doc 6."""
    if not y.index.equals(risks.index):
        raise ValueError("y and risks must share the same index and order")
    labels = y.to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    selected = select_advising_list(risks, capacity)
    n = len(y)
    budget = int(selected.sum())
    dropouts = int(y.sum())
    reached = int(y.loc[selected].sum())
    precision = reached / budget
    if dropouts == 0:
        recall = missed_rate = lift = None
    else:
        recall = reached / dropouts
        missed_rate = (dropouts - reached) / dropouts
        lift = precision / (dropouts / n)
    risk_values = risks.to_numpy(dtype=float)
    order = np.lexsort((risks.index.to_numpy(), -risk_values))
    cutoff_score = float(risk_values[order[budget - 1]])
    cutoff_tied = int((risk_values == cutoff_score).sum())
    cutoff_selected = int(((risk_values == cutoff_score) & selected.to_numpy(dtype=bool)).sum())
    return {
        "n": n, "budget": budget, "dropouts": dropouts, "reached": reached,
        "precision": precision, "recall": recall, "missed_rate": missed_rate,
        "lift": lift, "cutoff_score": cutoff_score,
        "cutoff_tied": cutoff_tied, "cutoff_selected": cutoff_selected,
    }


def group_metrics(
    y: pd.Series, selected: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """One row per group value, scored against the global advising list."""
    if not (y.index.equals(selected.index) and y.index.equals(groups.index)):
        raise ValueError("y, selected, and groups must have identical indexes")
    if selected.dtype != bool:
        raise ValueError("selected must be a boolean Series")
    rows = []
    for group in groups.unique():
        mask = groups.eq(group)
        group_y = y.loc[mask]
        group_selected = selected.loc[mask]
        n = len(group_y)
        dropouts = int(group_y.sum())
        selected_count = int(group_selected.sum())
        reached = int(group_y.loc[group_selected].sum())
        selection_rate = selected_count / n
        missed_rate = (dropouts - reached) / dropouts if dropouts > 0 else np.nan
        rows.append({
            "group": group, "n": n, "dropouts": dropouts, "selected": selected_count,
            "reached": reached, "selection_rate": selection_rate, "missed_rate": missed_rate,
        })
    return pd.DataFrame(rows).set_index("group").rename_axis("group")


class CourseBaseline:
    def __init__(self):
        self.course_risks_: dict[Any, float] = {}
        self.global_risk_: float | None = None
        self._fitted = False

    def fit(self, courses: pd.Series, y: pd.Series) -> "CourseBaseline":
        """Estimate each course's dropout fraction and the overall training mean."""
        if not courses.index.equals(y.index):
            raise ValueError("courses and y must share the same index")
        frame = pd.DataFrame({"course": courses.to_numpy(), "y": y.to_numpy()})
        self.course_risks_ = frame.groupby("course")["y"].mean().to_dict()
        self.global_risk_ = float(y.mean())
        self._fitted = True
        return self

    def predict_risk(self, courses: pd.Series) -> np.ndarray:
        """Course dropout fraction; unseen courses get the overall mean."""
        if not self._fitted:
            raise ValueError("Fit the baseline before predicting")
        return np.array(
            [self.course_risks_.get(course, self.global_risk_) for course in courses],
            dtype=float,
        )


def tune_id3(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_validation: pd.DataFrame, y_validation: pd.Series,
) -> tuple[ID3Preprocessor, ID3Classifier, dict[str, Any]]:
    """Fit the fixed ID3 configuration and score it once on validation rows."""
    numeric = [column for column in X_train if column in NUMERIC_COLUMNS]
    preprocessor = ID3Preprocessor(numeric, n_bins=3).fit(X_train)
    train_binned = preprocessor.transform(X_train)
    validation_binned = preprocessor.transform(X_validation)
    model = ID3Classifier(max_depth=3, min_samples_split=10).fit(train_binned, y_train)
    risks = pd.Series(model.predict_risk(validation_binned), index=X_validation.index)
    validation_precision = capacity_metrics(y_validation, risks)["precision"]
    settings = {
        "n_bins": 3, "max_depth": 3, "min_samples_split": 10,
        "validation_precision": validation_precision,
    }
    return preprocessor, model, settings


def tune_knn(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_validation: pd.DataFrame, y_validation: pd.Series,
    distance: str = "mixed",
) -> tuple[KNNPreprocessor | None, KNNClassifier, dict[str, Any]]:
    """Choose k by validation Precision@10%; equal scores prefer the larger k."""
    numeric = [column for column in X_train if column in NUMERIC_COLUMNS]
    categorical = [column for column in X_train if column in CATEGORICAL_COLUMNS]
    preprocessor = None
    train_input, validation_input = X_train, X_validation
    if distance == "mixed":
        preprocessor = KNNPreprocessor(numeric).fit(X_train)
        train_input = preprocessor.transform(X_train)
        validation_input = preprocessor.transform(X_validation)

    best_model = None
    best_precision = -1.0
    best_k = None
    for k in (3, 5, 11, 21, 51):
        model = KNNClassifier(
            k=k, distance=distance, numeric_columns=numeric,
            categorical_columns=categorical,
        ).fit(train_input, y_train)
        risks = pd.Series(model.predict_risk(validation_input), index=X_validation.index)
        precision = capacity_metrics(y_validation, risks)["precision"]
        if best_model is None or precision > best_precision or (precision == best_precision and k > best_k):
            best_model, best_precision, best_k = model, precision, k
    return preprocessor, best_model, {
        "k": best_k, "distance": distance,
        "validation_precision": best_precision,
    }


def bin_interval(preprocessor: ID3Preprocessor, column: str, bin_index: int) -> str:
    """Human-readable interval for a binned value, using the stored edges."""
    edges = preprocessor.bin_edges_.get(column, np.array([]))
    n_bins = len(edges) + 1
    lower = "-inf" if bin_index <= 0 else f"{edges[bin_index - 1]:.4g}"
    upper = "+inf" if bin_index >= n_bins - 1 else f"{edges[bin_index]:.4g}"
    return f"[{lower}, {upper})"


def run_experiments(df: pd.DataFrame, output_dir: Path, seed: int = SEED) -> None:
    """Run the full timeline experiment and save every required artifact."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(name: str, obj: Any) -> None:
        def default(value):
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            return str(value)
        (output_dir / name).write_text(json.dumps(obj, indent=2, default=default))

    # --- 1. Cohorts, features, one shared split --------------------------------
    X, y, enrolled_X = prepare_cohorts(df)
    columns_by_checkpoint = feature_sets(list(X.columns))
    split = split_row_ids(y, seed=seed)
    train_ids, validation_ids, test_ids = (split[name] for name in ("train", "validation", "test"))
    save_json("split_ids.json", {name: ids.tolist() for name, ids in split.items()})

    save_json("dataset_summary.json", {
        "n_records": int(len(df)),
        "n_predictors": int(len(X.columns)),
        "resolved_students": int(len(y)),
        "enrolled_students": int(len(enrolled_X)),
        "dropout_students": int(y.sum()),
        "graduate_students": int((y == 0).sum()),
        "dropout_prevalence_in_resolved": float(y.mean()),
        "train_validation_test": [len(train_ids), len(validation_ids), len(test_ids)],
        "duplicate_rows_including_target": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    })

    # Descriptive three-label outcome distribution per course (full file only).
    course_outcomes = (
        df.assign(_target=df["Target"]).groupby("Course")["Target"]
        .value_counts().unstack(fill_value=0)
        .reindex(columns=["Graduate", "Dropout", "Enrolled"], fill_value=0)
    )
    course_outcomes.to_csv(output_dir / "course_distributions.csv")

    # Column-level data checks: dtypes, observed value ranges, cardinality.
    data_checks = []
    for column in df.columns:
        series = df[column]
        if column == "Target":
            note = "three-level outcome label"
        elif not pd.api.types.is_numeric_dtype(series):
            note = "non-numeric column"
        else:
            note = ""
        data_checks.append({
            "column": column,
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique(dropna=True)),
            "n_missing": int(series.isna().sum()),
            "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) and series.notna().any() else None,
            "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) and series.notna().any() else None,
            "note": note,
        })
    pd.DataFrame(data_checks).to_csv(output_dir / "data_checks.csv", index=False)

    # Category frequencies for the nominal columns (doc 1: record what you find).
    frequency_rows = []
    for column in CATEGORICAL_COLUMNS:
        counts = df[column].value_counts(dropna=False).sort_index()
        for code, count in counts.items():
            frequency_rows.append({
                "column": column, "code": code, "count": int(count),
                "share": float(count / len(df)),
            })
    pd.DataFrame(frequency_rows).to_csv(output_dir / "categorical_frequencies.csv", index=False)

    y_train = y.loc[train_ids]
    y_validation = y.loc[validation_ids]
    y_test = y.loc[test_ids]

    # --- 2. Fit the six primary models ----------------------------------------
    fitted: dict[tuple[str, str], dict[str, Any]] = {}
    for checkpoint in CHECKPOINTS:
        columns = columns_by_checkpoint[checkpoint]
        X_train = X.loc[train_ids, columns]
        X_validation = X.loc[validation_ids, columns]
        X_test = X.loc[test_ids, columns]

        prep_id3, model_id3, settings_id3 = tune_id3(X_train, y_train, X_validation, y_validation)
        id3_test_risks = pd.Series(model_id3.predict_risk(prep_id3.transform(X_test)), index=X_test.index)
        fitted[("id3", checkpoint)] = {
            "preprocessor": prep_id3, "model": model_id3, "settings": settings_id3,
            "X_test": X_test, "test_risks": id3_test_risks,
            "test_metrics": capacity_metrics(y_test, id3_test_risks),
            "tree_stats": model_id3.tree_stats(),
        }
        id3_test_risks.to_csv(output_dir / f"id3_{checkpoint}_test_risks.csv",
                              header=True, index=True, index_label="row_id")
        save_json(f"id3_{checkpoint}_settings.json", {
            **settings_id3, "tree_stats": model_id3.tree_stats(),
            "validation_precision": settings_id3["validation_precision"],
        })

        prep_knn, model_knn, settings_knn = tune_knn(X_train, y_train, X_validation, y_validation, distance="mixed")
        knn_test_risks = pd.Series(model_knn.predict_risk(prep_knn.transform(X_test)), index=X_test.index)
        fitted[("knn", checkpoint)] = {
            "preprocessor": prep_knn, "model": model_knn, "settings": settings_knn,
            "X_test": X_test, "test_risks": knn_test_risks,
            "test_metrics": capacity_metrics(y_test, knn_test_risks),
        }
        knn_test_risks.to_csv(output_dir / f"knn_{checkpoint}_test_risks.csv",
                             header=True, index=True, index_label="row_id")
        save_json(f"knn_{checkpoint}_settings.json", settings_knn)

    # --- 3. Course-only baseline and random expectation on test ---------------
    course_train = X.loc[train_ids, "Course"]
    course_test = X.loc[test_ids, "Course"]
    baseline = CourseBaseline().fit(course_train, y_train)
    baseline_test_risks = pd.Series(baseline.predict_risk(course_test), index=X.loc[test_ids].index)
    baseline_metrics = capacity_metrics(y_test, baseline_test_risks)
    baseline_test_risks.to_csv(output_dir / "course_baseline_test_risks.csv",
                               header=True, index=True, index_label="row_id")

    n_test = len(y_test)
    dropouts_test = int(y_test.sum())
    budget_test = max(1, int(np.floor(0.10 * n_test)))
    random_expectation = {
        "expected_dropouts_reached": budget_test * dropouts_test / n_test,
        "expected_precision": dropouts_test / n_test,
        "expected_recall": budget_test / n_test,
        "expected_lift": 1.0,
    }
    baseline_table = pd.DataFrame([
        {"reference": "Course-only baseline (A1)", **baseline_metrics},
        {"reference": "Random selection expectation (A2)", **{
            "n": n_test, "budget": budget_test, "dropouts": dropouts_test,
            "reached": random_expectation["expected_dropouts_reached"],
            "precision": random_expectation["expected_precision"],
            "recall": random_expectation["expected_recall"],
            "lift": random_expectation["expected_lift"],
            "cutoff_score": None, "cutoff_tied": None, "cutoff_selected": None,
        }},
    ])
    baseline_table.to_csv(output_dir / "baseline_and_random.csv", index=False)

    # --- 4. Six-row timeline results table + two-panel plot -------------------
    timeline_rows = []
    for checkpoint in CHECKPOINTS:
        for kind in ("id3", "knn"):
            entry = fitted[(kind, checkpoint)]
            metrics = entry["test_metrics"]
            if kind == "id3":
                settings = "n_bins=3, max_depth=3, min_samples_split=10"
            else:
                settings = f"k={entry['settings']['k']}, mixed distance"
            timeline_rows.append({
                "checkpoint": checkpoint, "model": "ID3" if kind == "id3" else "kNN (mixed)",
                "n": metrics["n"], "B": metrics["budget"], "D": metrics["dropouts"],
                "dropouts_reached": metrics["reached"], "precision": metrics["precision"],
                "recall": metrics["recall"], "lift": metrics["lift"], "settings": settings,
                "cutoff_tied": metrics["cutoff_tied"], "cutoff_selected": metrics["cutoff_selected"],
            })
    timeline_table = pd.DataFrame(timeline_rows)
    timeline_table.to_csv(output_dir / "timeline_results.csv", index=False)
    save_json("timeline_results.json", timeline_rows)

    panel_order = {"enrollment": 0, "semester1": 1, "semester2": 2}
    id3_precision = [fitted[("id3", c)]["test_metrics"]["precision"] for c in CHECKPOINTS]
    knn_precision = [fitted[("knn", c)]["test_metrics"]["precision"] for c in CHECKPOINTS]
    id3_recall = [fitted[("id3", c)]["test_metrics"]["recall"] for c in CHECKPOINTS]
    knn_recall = [fitted[("knn", c)]["test_metrics"]["recall"] for c in CHECKPOINTS]
    x_positions = [panel_order[c] for c in CHECKPOINTS]
    checkpoint_labels = ["Enrollment", "Semester 1", "Semester 2"]

    fig, (ax_precision, ax_recall) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_precision.axhline(baseline_metrics["precision"], ls="--", color="#999999", label="Course baseline")
    ax_precision.axhline(random_expectation["expected_precision"], ls=":", color="#bbbbbb", label="Random expectation")
    ax_precision.plot(x_positions, id3_precision, marker="o", color="#1f77b4", label="ID3")
    ax_precision.plot(x_positions, knn_precision, marker="s", color="#d62728", label="kNN (mixed)")
    ax_precision.set_title("Precision@10%")
    ax_precision.set_xlabel("Checkpoint")
    ax_precision.legend(fontsize=8)
    ax_recall.axhline(baseline_metrics["recall"], ls="--", color="#999999", label="Course baseline")
    ax_recall.axhline(random_expectation["expected_recall"], ls=":", color="#bbbbbb", label="Random expectation")
    ax_recall.plot(x_positions, id3_recall, marker="o", color="#1f77b4", label="ID3")
    ax_recall.plot(x_positions, knn_recall, marker="s", color="#d62728", label="kNN (mixed)")
    ax_recall.set_title("Recall@10%")
    ax_recall.set_xlabel("Checkpoint")
    ax_recall.legend(fontsize=8)
    for axis in (ax_precision, ax_recall):
        axis.set_xticks(x_positions)
        axis.set_xticklabels(checkpoint_labels)
    fig.suptitle("Advising-list quality across prediction checkpoints (test set, 10% budget)")
    fig.tight_layout()
    fig.savefig(output_dir / "timeline_plot.png", dpi=150)
    plt.close(fig)

    # Checkpoint timeline figure (doc 3).
    fig_timeline, axis = plt.subplots(figsize=(8, 3.2))
    axis.axhline(0, color="#333333", lw=2)
    axis.plot([0, 2, 4, 6], [0, 0, 0, 0], "o", color="#333333")
    for number, (name, x) in enumerate(zip(
        ["Enrollment", "End of semester 1", "End of semester 2", "Normal completion outcome"],
        [0, 2, 4, 6],
    )):
        axis.annotate(name, (x, 0), xytext=(x, 0.25), ha="center", fontsize=9,
                      arrowprops=dict(arrowstyle="-", color="#999999"))
    axis.set_xlim(-0.6, 6.6)
    axis.set_ylim(-0.6, 0.7)
    axis.text(3, -0.45, "Prediction checkpoints (left) precede the resolved outcome label (right)",
              ha="center", fontsize=8, color="#666666")
    axis.axis("off")
    axis.set_title("Checkpoint timeline vs. outcome horizon")
    fig_timeline.tight_layout()
    fig_timeline.savefig(output_dir / "checkpoint_timeline.png", dpi=150)
    plt.close(fig_timeline)

    # --- 5. kNN raw vs mixed (A3) and redundancy ablation (A4) -----------------
    columns_s2 = columns_by_checkpoint["semester2"]
    X_train_s2 = X.loc[train_ids, columns_s2]
    X_validation_s2 = X.loc[validation_ids, columns_s2]
    X_test_s2 = X.loc[test_ids, columns_s2]

    prep_raw, model_raw, settings_raw = tune_knn(X_train_s2, y_train, X_validation_s2, y_validation, distance="raw")
    raw_test_risks = pd.Series(model_raw.predict_risk(X_test_s2), index=X_test_s2.index)
    raw_metrics = capacity_metrics(y_test, raw_test_risks)
    raw_test_risks.to_csv(output_dir / "knn_raw_semester2_test_risks.csv",
                          header=True, index=True, index_label="row_id")
    mixed_s2 = fitted[("knn", "semester2")]
    raw_vs_mixed = pd.DataFrame([
        {"distance": "raw (A3)", "settings": f"k={settings_raw['k']}",
         "validation_precision": settings_raw["validation_precision"],
         "test_precision": raw_metrics["precision"], "test_recall": raw_metrics["recall"],
         "dropouts_reached": raw_metrics["reached"]},
        {"distance": "mixed (P6)", "settings": f"k={mixed_s2['settings']['k']}",
         "validation_precision": mixed_s2["settings"]["validation_precision"],
         "test_precision": mixed_s2["test_metrics"]["precision"],
         "test_recall": mixed_s2["test_metrics"]["recall"],
         "dropouts_reached": mixed_s2["test_metrics"]["reached"]},
    ])
    raw_vs_mixed.to_csv(output_dir / "knn_raw_vs_mixed.csv", index=False)

    reduced_columns = [c for c in columns_s2 if c not in ("International", "Curricular units 1st sem (approved)")]
    X_train_red = X.loc[train_ids, reduced_columns]
    X_validation_red = X.loc[validation_ids, reduced_columns]
    X_test_red = X.loc[test_ids, reduced_columns]
    prep_red, model_red, settings_red = tune_knn(X_train_red, y_train, X_validation_red, y_validation, distance="mixed")
    reduced_test_risks = pd.Series(model_red.predict_risk(prep_red.transform(X_test_red)), index=X_test_red.index)
    reduced_metrics = capacity_metrics(y_test, reduced_test_risks)
    reduced_test_risks.to_csv(output_dir / "knn_reduced_semester2_test_risks.csv",
                              header=True, index=True, index_label="row_id")

    full_selection = select_advising_list(mixed_s2["test_risks"])
    reduced_selection = select_advising_list(reduced_test_risks)
    overlap = int((full_selection & reduced_selection).sum())
    ablation_table = pd.DataFrame([
        {"run": "P6 full mixed", "k": mixed_s2["settings"]["k"],
         "test_precision": mixed_s2["test_metrics"]["precision"],
         "test_recall": mixed_s2["test_metrics"]["recall"],
         "dropouts_reached": mixed_s2["test_metrics"]["reached"]},
        {"run": "A4 reduced mixed (drop International, 1st-sem approved)",
         "k": settings_red["k"],
         "test_precision": reduced_metrics["precision"], "test_recall": reduced_metrics["recall"],
         "dropouts_reached": reduced_metrics["reached"]},
    ])
    ablation_table.to_csv(output_dir / "knn_reduced_ablation.csv", index=False)
    save_json("knn_reduced_ablation_summary.json", {
        "removed_features": ["International", "Curricular units 1st sem (approved)"],
        "overlap_with_full_selection": overlap, "budget": budget_test,
        "full": {"k": mixed_s2["settings"]["k"], **mixed_s2["test_metrics"]},
        "reduced": {"k": settings_red["k"], **reduced_metrics},
    })

    # Redundancy evidence: correlations and a Nacionality x International cross-tab.
    pearson_approved = float(X["Curricular units 1st sem (approved)"]
                            .corr(X["Curricular units 2nd sem (approved)"]))
    pearson_codes = float(X["Nacionality"].astype(float).corr(X["International"].astype(float)))
    crosstab = pd.crosstab(X["Nacionality"], X["International"])
    redundancy = {
        "corr_1stsem_2ndsem_approved": pearson_approved,
        "corr_nacionality_international_on_codes": pearson_codes,
        "nacionality_international_crosstab": crosstab.to_dict(),
        "note": "A Pearson correlation on arbitrary category codes is not a numeric similarity.",
    }
    save_json("redundancy_evidence.json", redundancy)
    crosstab.to_csv(output_dir / "nacionality_international_crosstab.csv")

    # Neighbour ordering for one test student under raw vs mixed distance (doc 5).
    focus_id = int(mixed_s2["test_risks"].idxmax())
    focus_row = X_test_s2.loc[[focus_id]]
    raw_dists, raw_ids = model_raw.kneighbors(focus_row, n_neighbors=5)
    mixed_dists, mixed_ids = mixed_s2["model"].kneighbors(mixed_s2["preprocessor"].transform(focus_row), n_neighbors=5)
    neighbor_comparison = {
        "student_row_id": focus_id,
        "raw_neighbors": [
            {"row_id": int(rid), "distance": float(dist),
             "outcome": int(y.loc[rid]) if rid in y.index else None}
            for dist, rid in zip(raw_dists[0], raw_ids[0])
        ],
        "mixed_neighbors": [
            {"row_id": int(rid), "distance": float(dist),
             "outcome": int(y.loc[rid]) if rid in y.index else None}
            for dist, rid in zip(mixed_dists[0], mixed_ids[0])
        ],
    }
    save_json("knn_neighbor_comparison.json", neighbor_comparison)

    # --- 6. Gender removal (A5, A6) ------------------------------------------
    columns_enrollment = columns_by_checkpoint["enrollment"]
    no_gender_columns = [c for c in columns_enrollment if c != "Gender"]
    X_train_enr = X.loc[train_ids, no_gender_columns]
    X_validation_enr = X.loc[validation_ids, no_gender_columns]
    X_test_enr = X.loc[test_ids, no_gender_columns]

    prep_id3_ng, model_id3_ng, settings_id3_ng = tune_id3(X_train_enr, y_train, X_validation_enr, y_validation)
    id3_ng_risks = pd.Series(model_id3_ng.predict_risk(prep_id3_ng.transform(X_test_enr)), index=X_test_enr.index)
    id3_ng_metrics = capacity_metrics(y_test, id3_ng_risks)

    prep_knn_ng, model_knn_ng, settings_knn_ng = tune_knn(X_train_enr, y_train, X_validation_enr, y_validation, distance="mixed")
    knn_ng_risks = pd.Series(model_knn_ng.predict_risk(prep_knn_ng.transform(X_test_enr)), index=X_test_enr.index)
    knn_ng_metrics = capacity_metrics(y_test, knn_ng_risks)
    gender_removal = pd.DataFrame([
        {"run": "P1 ID3 (with Gender)", "test_precision": fitted[("id3", "enrollment")]["test_metrics"]["precision"],
         "test_recall": fitted[("id3", "enrollment")]["test_metrics"]["recall"]},
        {"run": "A5 ID3 (Gender removed)", "test_precision": id3_ng_metrics["precision"],
         "test_recall": id3_ng_metrics["recall"]},
        {"run": "P4 kNN (with Gender)", "test_precision": fitted[("knn", "enrollment")]["test_metrics"]["precision"],
         "test_recall": fitted[("knn", "enrollment")]["test_metrics"]["recall"]},
        {"run": "A6 kNN (Gender removed)", "test_precision": knn_ng_metrics["precision"],
         "test_recall": knn_ng_metrics["recall"]},
    ])
    gender_removal.to_csv(output_dir / "gender_removal_metrics.csv", index=False)
    save_json("gender_removal_settings.json", {
        "A5_id3": settings_id3_ng, "A6_knn": settings_knn_ng,
    })

    # --- 7. Paired explanations (three test students) -------------------------
    p3 = fitted[("id3", "semester2")]
    p6 = fitted[("knn", "semester2")]
    p3_risks = p3["test_risks"]

    def pick_closest_to_half(series: pd.Series, excluded: set[int]) -> int:
        candidates = series.drop(index=list(excluded), errors="ignore")
        distance = (candidates - 0.5).abs()
        minimum = distance.min()
        return int(distance[distance == minimum].index.min())

    picked: list[int] = []
    student1 = int(p3_risks[p3_risks == p3_risks.max()].index.min())
    picked.append(student1)
    student2 = pick_closest_to_half(p3_risks, set(picked))
    picked.append(student2)
    remaining_risks = p3_risks.drop(index=picked, errors="ignore")
    minimum = remaining_risks.min()
    student3 = int(remaining_risks[remaining_risks == minimum].index.min())
    explanation_students = picked + [student3]
    save_json("explanation_students.json", {
        "rule": "P3 semester-2 ID3 test risks: (1) highest, (2) closest to 0.5, (3) lowest; smaller row ID breaks ties",
        "students": explanation_students,
    })

    explanations = []
    scores2_columns = columns_by_checkpoint["semester2"]
    for student in explanation_students:
        raw_row = X.loc[test_ids, scores2_columns].loc[[student]]
        binned_row = p3["preprocessor"].transform(raw_row).iloc[0]
        path = p3["model"].decision_path(binned_row)
        id3_path = []
        for step in path:
            feature = step["feature"]
            if feature is not None and feature in NUMERIC_COLUMNS:
                interval = bin_interval(p3["preprocessor"], feature, step["value"])
                display = f"bin {step['value']} ({interval})"
            else:
                display = step["value"]
            counts = step["counts"]
            total = counts[0] + counts[1]
            id3_path.append({
                "feature": feature, "value": step["value"], "display_value": display,
                "counts_graduate": counts[0], "counts_dropout": counts[1],
                "risk": (counts[1] / total) if total else None,
                "fallback": step["fallback"],
            })
        id3_risk = float(p3["model"].predict_risk(p3["preprocessor"].transform(raw_row))[0])

        scaled_row = p6["preprocessor"].transform(raw_row)
        five_dists, five_ids = p6["model"].kneighbors(scaled_row, n_neighbors=5)
        knn_k = p6["settings"]["k"]
        all_dists, all_ids = p6["model"].kneighbors(scaled_row)  # uses tuned k
        full_vote_labels = p6["model"].y_train_.reindex(all_ids[0]).to_numpy()
        knn_risk = float(p6["model"].predict_risk(scaled_row)[0])
        neighbors5 = []
        comparisons = []
        student_raw = raw_row.iloc[0]
        for dist, rid in zip(five_dists[0], five_ids[0]):
            rid = int(rid)
            neighbors5.append({
                "row_id": rid, "distance": float(dist),
                "outcome": int(y.loc[rid]) if rid in y.index else None,
            })
            neighbor_raw = X.loc[rid]
            mismatches = [c for c in CATEGORICAL_COLUMNS
                          if c in student_raw.index and student_raw[c] != neighbor_raw[c]]
            numeric_diffs = sorted(
                ({"feature": c, "student": float(student_raw[c]),
                  "neighbor": float(neighbor_raw[c]),
                  "absolute_difference": abs(float(student_raw[c]) - float(neighbor_raw[c]))}
                 for c in NUMERIC_COLUMNS
                 if c in student_raw.index and float(student_raw[c]) != float(neighbor_raw[c])),
                key=lambda d: d["absolute_difference"], reverse=True,
            )
            comparisons.append({
                "row_id": rid,
                "n_categorical_matches": sum(1 for c in CATEGORICAL_COLUMNS
                                             if c in student_raw.index and student_raw[c] == neighbor_raw[c]),
                "categorical_mismatches": mismatches,
                "largest_numeric_differences": numeric_diffs[:4],
            })
        explanations.append({
            "student_row_id": student,
            "true_outcome": int(y_test.loc[student]),
            "id3": {"path": id3_path, "risk": id3_risk},
            "knn": {
                "tuned_k": knn_k, "five_neighbors": neighbors5, "comparisons": comparisons,
                "full_vote_dropout_fraction": float(full_vote_labels.mean()),
                "risk_over_tuned_k": knn_risk,
            },
        })
    save_json("explanations.json", explanations)

    # --- 8. Missed-dropout audits (P1, P4) and gender-removal audits ----------
    gender_test = X.loc[test_ids, "Gender"]
    scholarship_test = X.loc[test_ids, "Scholarship holder"]
    fees_test = X.loc[test_ids, "Tuition fees up to date"]

    def audit_table(risks: pd.Series, attribute: pd.Series) -> pd.DataFrame:
        selected = select_advising_list(risks)
        table = group_metrics(y_test, selected, attribute).reset_index()
        return table

    fairness_records = {}
    selection_models = {
        "P1_id3": fitted[("id3", "enrollment")]["test_risks"],
        "P4_knn": fitted[("knn", "enrollment")]["test_risks"],
        "A5_id3_no_gender": id3_ng_risks,
        "A6_knn_no_gender": knn_ng_risks,
    }
    attributes = {"Gender": gender_test, "Scholarship holder": scholarship_test,
                  "Tuition fees up to date": fees_test}
    all_fairness = []
    for model_name, risks in selection_models.items():
        for attribute_name, attribute in attributes.items():
            table = audit_table(risks, attribute)
            table.insert(0, "model", model_name)
            table.insert(1, "attribute", attribute_name)
            all_fairness.append(table)
            # Missed-rate difference (coded 1 minus coded 0).
            indexed = table.set_index("group")
            groups_available = set(indexed.index.tolist())
            if 1 in groups_available and 0 in groups_available:
                difference = float(indexed.loc[1, "missed_rate"] - indexed.loc[0, "missed_rate"])
            else:
                difference = None
            fairness_records[f"{model_name}|{attribute_name}"] = {
                "missed_rate_difference_g1_minus_g0": difference,
            }
    fairness_table = pd.concat(all_fairness, ignore_index=True)
    fairness_table.to_csv(output_dir / "fairness_group_tables.csv", index=False)
    save_json("fairness_missed_rate_differences.json", fairness_records)

    # Gender-gap change before and after removal.
    def gender_gap(risks: pd.Series) -> Any:
        selected = select_advising_list(risks)
        table = group_metrics(y_test, selected, gender_test)
        if 1 in table.index and 0 in table.index and pd.notna(table.loc[1, "missed_rate"]) and pd.notna(table.loc[0, "missed_rate"]):
            return float(table.loc[1, "missed_rate"] - table.loc[0, "missed_rate"])
        return None

    gender_gap_comparison = {
        "P4_knn_with_gender": gender_gap(fitted[("knn", "enrollment")]["test_risks"]),
        "A6_knn_no_gender": gender_gap(knn_ng_risks),
        "P1_id3_with_gender": gender_gap(fitted[("id3", "enrollment")]["test_risks"]),
        "A5_id3_no_gender": gender_gap(id3_ng_risks),
        "interpretation": "missed-dropout rate for Gender=1 (male) minus Gender=0 (female)",
    }
    save_json("gender_gap_comparison.json", gender_gap_comparison)

    # --- Consolidated baseline + ablation table (A1-A6, test set) -----------------
    ablation_table = pd.DataFrame([
        {"run": "A1 Course-only baseline",
         "chosen_settings": "train-course dropout fraction", **baseline_metrics},
        {"run": "A2 Random-selection expectation",
         "chosen_settings": "none (formula, not a trained model)",
         "n": n_test, "budget": budget_test, "dropouts": dropouts_test,
         "reached": random_expectation["expected_dropouts_reached"],
         "precision": random_expectation["expected_precision"],
         "recall": random_expectation["expected_recall"],
         "missed_rate": 1 - random_expectation["expected_recall"],
         "lift": random_expectation["expected_lift"],
         "cutoff_score": None, "cutoff_tied": None, "cutoff_selected": None},
        {"run": "A3 kNN raw features (semester 2)",
         "chosen_settings": f"k={settings_raw['k']} (raw)", **raw_metrics},
        {"run": "A4 kNN reduced features (semester 2)",
         "chosen_settings": "k={} (mixed; International and 1st-sem approved removed)".format(settings_red["k"]),
         **reduced_metrics},
        {"run": "A5 ID3 no Gender (enrollment)",
         "chosen_settings": f"fixed (depth={settings_id3_ng['max_depth']}, bins={settings_id3_ng['n_bins']})",
         **id3_ng_metrics},
        {"run": "A6 kNN no Gender (enrollment)",
         "chosen_settings": f"k={settings_knn_ng['k']} (mixed)", **knn_ng_metrics},
    ])
    ablation_table.to_csv(output_dir / "ablations_A1_A6.csv", index=False)

    # --- Test risk scores and the exported selection mask for every trained run ---
    selection_sources = {
        f"P_id3_{cp}": fitted[("id3", cp)]["test_risks"] for cp in CHECKPOINTS
    } | {
        f"P_knn_{cp}": fitted[("knn", cp)]["test_risks"] for cp in CHECKPOINTS
    } | {
        "A1_course_baseline": baseline_test_risks,
        "A3_knn_raw_semester2": raw_test_risks,
        "A4_knn_reduced_semester2": reduced_test_risks,
        "A5_id3_no_gender": id3_ng_risks,
        "A6_knn_no_gender": knn_ng_risks,
    }
    selection_frames = []
    for run_name, risks in selection_sources.items():
        mask = select_advising_list(risks)
        frame = pd.DataFrame({
            "run": run_name, "row_id": risks.index, "risk": risks.to_numpy(),
            "selected": mask.to_numpy(), "true_outcome": y_test.loc[risks.index].to_numpy(),
        })
        frame.to_csv(output_dir / f"selection_{run_name}.csv", index=False)
        selection_frames.append(frame)
    pd.concat(selection_frames, ignore_index=True).to_csv(
        output_dir / "test_selections_all_runs.csv", index=False)

    # Every kNN run's selected k and validation Precision@10% (doc 5 evidence).
    knn_tuning = pd.DataFrame([
        {"run": name, "distance": settings["distance"], "k": settings["k"],
         "validation_precision": settings["validation_precision"]}
        for name, settings in (
            ("P4 kNN enrollment", fitted[("knn", "enrollment")]["settings"]),
            ("P5 kNN semester 1", fitted[("knn", "semester1")]["settings"]),
            ("P6 kNN semester 2", fitted[("knn", "semester2")]["settings"]),
            ("A3 kNN raw semester 2", settings_raw),
            ("A4 kNN reduced semester 2", settings_red),
            ("A6 kNN no Gender enrollment", settings_knn_ng),
        )
    ])
    knn_tuning.to_csv(output_dir / "knn_tuning_summary.csv", index=False)

    # --- 9. Data detective ----------------------------------------------------
    economic_columns = ["Unemployment rate", "Inflation rate", "GDP"]
    labeled_only = X.copy()
    labeled_only = labeled_only.assign(_y=y.to_numpy())
    triple_summary = labeled_only.groupby(economic_columns).agg(
        n=("_y", "size"), dropouts=("_y", "sum"),
    )
    triple_summary["dropout_rate"] = triple_summary["dropouts"] / triple_summary["n"]
    triple_summary["graduates"] = triple_summary["n"] - triple_summary["dropouts"]
    triple_summary = triple_summary.reset_index()
    triple_summary.to_csv(output_dir / "economic_triples.csv", index=False)
    save_json("economic_triples_summary.json", {
        "n_distinct_triples": int(len(triple_summary)),
        "table": triple_summary.to_dict(orient="records"),
    })

    grade_columns = {
        "semester1": "Curricular units 1st sem (grade)",
        "semester2": "Curricular units 2nd sem (grade)",
    }
    zero_grade_summary = []
    for semester, column in grade_columns.items():
        grades = df[column]
        zero_summary = {
            "semester": semester, "column": column,
            "zero_count": int((grades == 0).sum()),
            "mean_grade": float(grades.mean()), "median_grade": float(grades.median()),
            "nonzero_mean": float(grades[grades > 0].mean()),
        }
        zero_grade_summary.append(zero_summary)
    pd.DataFrame(zero_grade_summary).to_csv(output_dir / "zero_grade_summary.csv", index=False)

    crosstabs = {}
    for semester, prefix, grade_column in [
        ("semester1", "Curricular units 1st sem", grade_columns["semester1"]),
        ("semester2", "Curricular units 2nd sem", grade_columns["semester2"]),
    ]:
        for measure in ["enrolled", "approved", "without evaluations"]:
            column = f"{prefix} ({measure})"
            df["_zero"] = (df[grade_column] == 0)
            table = pd.crosstab(df[column], df["_zero"])
            table.columns = ["nonzero_grade" if not col else "zero_grade" for col in table.columns]
            crosstabs[f"{semester}_{measure.replace(' ', '_')}"] = table.astype(int).to_dict()
    df.drop(columns="_zero", inplace=True)
    save_json("zero_grade_crosstabs.json", crosstabs)

    # Enrolled exploratory scoring with frozen semester-2 models.
    enrolled_columns = columns_by_checkpoint["semester2"]
    enrolled_X_s2 = enrolled_X[enrolled_columns]
    enrolled_id3_risks = pd.Series(
        p3["model"].predict_risk(p3["preprocessor"].transform(enrolled_X_s2)),
        index=enrolled_X_s2.index,
    )
    enrolled_knn_risks = pd.Series(
        p6["model"].predict_risk(p6["preprocessor"].transform(enrolled_X_s2)),
        index=enrolled_X_s2.index,
    )
    enrolled_out = pd.DataFrame({
        "id3_risk": enrolled_id3_risks,
        "knn_risk": enrolled_knn_risks,
    })
    enrolled_out.index.name = "row_id"
    enrolled_out.to_csv(output_dir / "enrolled_students_risk_scores.csv")

    def describe(series: pd.Series) -> dict[str, float]:
        return {
            "count": int(series.shape[0]), "mean": float(series.mean()),
            "std": float(series.std(ddof=1)), "min": float(series.min()),
            "p25": float(series.quantile(0.25)), "median": float(series.median()),
            "p75": float(series.quantile(0.75)), "max": float(series.max()),
        }

    enrolled_summary = {
        "enrolled_id3": describe(enrolled_id3_risks),
        "enrolled_knn": describe(enrolled_knn_risks),
        "test_id3_P3": describe(p3["test_risks"]),
        "test_knn_P6": describe(p6["test_risks"]),
        "note": "Enrolled students have unresolved outcomes; no precision/recall/accuracy is computed.",
    }
    save_json("enrolled_vs_test_risk_summary.json", enrolled_summary)

    # Enrolled risk histogram next to the labeled test distribution.
    fig_enrolled, axis = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 21)
    axis[0].hist(enrolled_id3_risks, bins=bins, alpha=0.7, color="#2ca02c")
    axis[0].set_title("Enrolled students - ID3 (P3) risk")
    axis[0].set_xlabel("dropout risk")
    axis[0].set_ylabel("students")
    axis[1].hist(p3["test_risks"], bins=bins, alpha=0.7, color="#1f77b4")
    axis[1].set_title("Labeled test students - ID3 (P3) risk")
    axis[1].set_xlabel("dropout risk")
    fig_enrolled.suptitle("Enrolled vs. resolved-test risk distributions (semester 2, ID3)")
    fig_enrolled.tight_layout()
    fig_enrolled.savefig(output_dir / "enrolled_risk_histogram.png", dpi=150)
    plt.close(fig_enrolled)

    save_json("run_matrix_summary.json", {
        "timeline": timeline_rows,
        "baseline": baseline_metrics,
        "random_expectation": random_expectation,
        "raw_vs_mixed": raw_vs_mixed.to_dict(orient="records"),
        "reduced_ablation_overlap": overlap,
        "gender_removal_metrics": gender_removal.to_dict(orient="records"),
        "gender_gap": gender_gap_comparison,
        "enrolled_summary": enrolled_summary,
        "economic_triples_count": int(len(triple_summary)),
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inspect", action="store_true", help="Inspect the CSV without running TODOs")
    args = parser.parse_args(argv)
    df = load_dataset(args.data)
    if args.inspect:
        print(f"Rows: {len(df)}; predictors: {len(df.columns) - 1}")
        print(df["Target"].value_counts().to_string())
        print(f"Missing cells: {int(df.isna().sum().sum())}")
        return 0
    try:
        run_experiments(df, args.output, args.seed)
    except NotImplementedError as exc:
        print(f"Starter TODO: {exc}. See docs/README.md.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
