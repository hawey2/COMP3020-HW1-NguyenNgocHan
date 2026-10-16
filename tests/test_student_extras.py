"""Additional checks for the student implementation (not part of the supplied 46).

The supplied suite does not exercise the fitting/tuning helpers or the full
``run_experiments`` pipeline, so these tests run the whole pipeline on a small
synthetic file and check the artifacts it must save, plus the helper invariants
used throughout (bin-interval translation, budget/tie-break rules).
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main import (
    CHECKPOINTS, CourseBaseline, ID3Preprocessor, bin_interval, capacity_metrics,
    feature_sets, run_experiments, select_advising_list,
)

REAL_DATASET = Path(__file__).resolve().parents[1] / "dataset.csv"


def _load_real_columns() -> list[str]:
    """Original column order (excluding Target) from the supplied dataset."""
    df = pd.read_csv(REAL_DATASET)
    return [c for c in df.columns if c != "Target"]


def _tiny_dataset(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic file with the real schema and a learnable dropout signal."""
    rng = np.random.default_rng(seed)
    columns = _load_real_columns()
    size = n // 2
    # Two separable blobs plus noise, so tree/kNN/baseline are all exercised.
    base_low = rng.normal(0.0, 1.0, size=(size, len(columns)))
    base_high = rng.normal(1.5, 1.0, size=(size, len(columns)))
    values = np.vstack([base_low, base_high])
    labels = np.array(["Graduate"] * size + ["Dropout"] * size, dtype=object)
    # Shuffle so classes are not contiguous; row_id is assigned by row position
    # on write, exactly as load_dataset does for the real file.
    order = rng.permutation(len(values))
    df = pd.DataFrame(np.round(values[order], 4), columns=columns)
    labels = labels[order]
    # Leave a share unresolved to exercise the Enrolled path.
    labels[rng.random(len(df)) < 0.15] = "Enrolled"
    df["Target"] = labels
    # Keep at least two levels in a few categorical columns for tie-breaks.
    for column in ("Course", "Nacionality", "Gender"):
        df[column] = np.where(df[column] > df[column].median(), 1, 0).astype(int)
    return df


class TestBinInterval(unittest.TestCase):
    """Checks for the human-readable bin-interval translation (doc 4)."""

    def setUp(self):
        self.columns = _load_real_columns()
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            rng.normal(size=(200, len(self.columns))), columns=self.columns,
        )
        df.index.name = "row_id"
        numeric = [c for c in self.columns if c in ("Admission grade", "Age at enrollment")]
        self.numeric = numeric
        self.preprocessor = ID3Preprocessor(numeric, n_bins=3).fit(df)

    def test_intervals_are_well_formed_and_open_ended_at_the_ends(self):
        for column in self.numeric:
            edges = self.preprocessor.bin_edges_[column]
            n_bins = len(edges) + 1
            first = bin_interval(self.preprocessor, column, 0)
            last = bin_interval(self.preprocessor, column, n_bins - 1)
            self.assertTrue(first.startswith("[-inf,"), first)
            self.assertTrue(last.endswith("+inf)"), last)
            for inner in range(1, n_bins - 1):
                interval = bin_interval(self.preprocessor, column, inner)
                self.assertTrue(interval.startswith("["), interval)
                self.assertTrue(interval.endswith(")"), interval)

    def test_intervals_tile_the_real_line_without_gaps_or_overlap(self):
        def bounds(interval):
            parts = [p.strip() for p in interval[1:-1].split(",")]
            return parts[0], parts[1]

        for column in self.numeric:
            edges = self.preprocessor.bin_edges_[column]
            n_bins = len(edges) + 1
            intervals = [bin_interval(self.preprocessor, column, b) for b in range(n_bins)]
            for index in range(1, n_bins):
                lower_open = bounds(intervals[index])[0]
                upper_close = bounds(intervals[index - 1])[1]
                self.assertEqual(lower_open, upper_close)


class TestSelectAdvisingList(unittest.TestCase):
    """Budget, ordering, and tie-break invariants (doc 6)."""

    def test_budget_is_floor_of_ten_percent_and_at_least_one(self):
        for n, expected in ((726, 72), (100, 10), (10, 1), (5, 1), (9, 1)):
            risks = pd.Series(np.linspace(0, 1, n), index=range(n))
            self.assertEqual(select_advising_list(risks).sum(), expected)

    def test_ties_are_selected_in_ascending_row_id_order(self):
        # 20 students: budget = 2. The top 18 are tied; ids 2 and 3 win the tie.
        index = list(range(19, -1, -1))
        risks = pd.Series([1.0] * 18 + [0.0, 0.0], index=index)
        selected = select_advising_list(risks)
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(sorted(int(i) for i in selected[selected].index), [2, 3])

    def test_selection_uses_a_single_global_budget(self):
        # 10% of ten students is exactly one, regardless of score clustering.
        for risks in (pd.Series([0.1, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], index=range(10)),
                      pd.Series(np.linspace(0, 1, 10), index=range(10))):
            self.assertEqual(select_advising_list(risks).sum(), 1)

    def test_tie_counts_are_consistent_with_the_report_keys(self):
        risks = pd.Series([0.9] * 15 + [0.5] * 5, index=range(20))
        mask = select_advising_list(risks)
        cutoff = float(risks[mask].min())
        tied = int((risks == cutoff).sum())
        metrics = capacity_metrics(pd.Series([1] * 15 + [0] * 5, index=range(20)), risks)
        self.assertEqual(tied, metrics["cutoff_tied"])
        self.assertEqual(int(mask[risks == cutoff].sum()), metrics["cutoff_selected"])
        self.assertLessEqual(metrics["cutoff_selected"], metrics["cutoff_tied"])
        self.assertEqual(int(mask.sum()), metrics["budget"])


class TestRedundancyAndBaseline(unittest.TestCase):
    """Course baseline behaviour and feature-set nesting (docs 3, 5)."""

    def test_unseen_course_falls_back_to_the_training_prevalence(self):
        train = pd.Series(["a", "a", "b", "c"], index=[0, 1, 2, 3])
        labels = pd.Series([1, 0, 1, 0], index=[0, 1, 2, 3])
        model = CourseBaseline().fit(train, labels)
        risks = np.asarray(model.predict_risk(pd.Series(["a", "zzz"], index=[10, 11])))
        self.assertAlmostEqual(risks[0], 0.5)
        self.assertAlmostEqual(risks[1], 0.5)  # unseen -> training prevalence
        self.assertTrue(((risks >= 0) & (risks <= 1)).all())

    def test_feature_sets_are_nested_and_cover_all_predictors(self):
        all_columns = _load_real_columns()
        sets = feature_sets(all_columns)
        # Columns keep their original order, so nesting is by membership.
        self.assertTrue(set(sets["enrollment"]).issubset(set(sets["semester1"])))
        self.assertTrue(set(sets["semester1"]).issubset(set(sets["semester2"])))
        self.assertEqual(set(sets["semester2"]), set(all_columns))
        # Future semester columns must not leak into earlier checkpoints.
        self.assertTrue(all("2nd sem" not in c for c in sets["semester1"]))
        self.assertTrue(all("1st sem" not in c and "2nd sem" not in c for c in sets["enrollment"]))


class TestPipelineArtifacts(unittest.TestCase):
    """End-to-end check that every required artifact is written (doc 10)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.out = cls.tmp / "results"
        cls.data = cls.tmp / "tiny.csv"
        _tiny_dataset().to_csv(cls.data, index=False)
        run_experiments(cls._load(cls.data), cls.out, seed=42)

        with open(cls.out / "split_ids.json") as fh:
            split = json.load(fh)
        cls.split = split
        cls.primary = pd.read_csv(cls.out / "timeline_results.csv")

    @staticmethod
    def _load(path):
        from main import load_dataset
        return load_dataset(path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_required_artifact_exists(self):
        required = [
            "split_ids.json", "dataset_summary.json", "course_distributions.csv",
            "data_checks.csv", "categorical_frequencies.csv",
            "timeline_results.csv", "timeline_results.json", "timeline_plot.png",
            "checkpoint_timeline.png",
            "course_baseline_test_risks.csv", "baseline_and_random.csv",
            "ablations_A1_A6.csv", "test_selections_all_runs.csv",
            "knn_tuning_summary.csv",
            "knn_raw_vs_mixed.csv", "knn_reduced_ablation.csv",
            "knn_reduced_ablation_summary.json",
            "redundancy_evidence.json", "nacionality_international_crosstab.csv",
            "knn_neighbor_comparison.json",
            "gender_removal_metrics.csv", "gender_removal_settings.json",
            "fairness_group_tables.csv", "fairness_missed_rate_differences.json",
            "gender_gap_comparison.json",
            "explanation_students.json", "explanations.json",
            "economic_triples.csv", "economic_triples_summary.json",
            "zero_grade_summary.csv", "zero_grade_crosstabs.json",
            "enrolled_risk_histogram.png", "enrolled_vs_test_risk_summary.json",
            "enrolled_students_risk_scores.csv",
        ]
        for name in required:
            self.assertTrue((self.out / name).exists(), f"missing artifact: {name}")
        # A settings/selection/risk file per primary and ablation run.
        for checkpoint in CHECKPOINTS:
            for model in ("id3", "knn"):
                self.assertTrue((self.out / f"{model}_{checkpoint}_settings.json").exists())
                self.assertTrue((self.out / f"{model}_{checkpoint}_test_risks.csv").exists())
        for run in ("P_id3_enrollment", "P_id3_semester1", "P_id3_semester2",
                    "P_knn_enrollment", "P_knn_semester1", "P_knn_semester2",
                    "A1_course_baseline", "A3_knn_raw_semester2",
                    "A4_knn_reduced_semester2", "A5_id3_no_gender", "A6_knn_no_gender"):
            self.assertTrue((self.out / f"selection_{run}.csv").exists(),
                            f"missing selection file for {run}")

    def test_splits_are_disjoint_resolved_row_ids(self):
        train, validation, test = (set(self.split[k]) for k in ("train", "validation", "test"))
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))
        self.assertEqual(len(train | validation | test), len(train) + len(validation) + len(test))

    def test_six_row_timeline_table(self):
        self.assertEqual(len(self.primary), 6)
        self.assertEqual(self.primary["model"].tolist(),
                         ["ID3", "kNN (mixed)", "ID3", "kNN (mixed)", "ID3", "kNN (mixed)"])
        self.assertEqual(self.primary["checkpoint"].tolist(),
                         ["enrollment", "enrollment", "semester1", "semester1",
                          "semester2", "semester2"])
        self.assertTrue((self.primary["B"] == self.primary["B"].iloc[0]).all())
        self.assertTrue(self.primary["precision"].between(0, 1).all())
        self.assertTrue(self.primary["recall"].between(0, 1).all())

    def test_exported_selections_match_the_reported_cutoffs(self):
        selections = pd.read_csv(self.out / "test_selections_all_runs.csv")
        for _, row in self.primary.iterrows():
            model = "id3" if row["model"] == "ID3" else "knn"
            run = f"P_{model}_{row['checkpoint']}"
            selected = selections[selections["run"] == run]
            self.assertEqual(int(selected["selected"].sum()), int(row["B"]))
            self.assertEqual(int(selected.loc[selected["selected"], "true_outcome"].sum()),
                             int(row["dropouts_reached"]))

    def test_selection_files_agree_with_the_consolidated_table(self):
        one_run = pd.read_csv(self.out / "selection_A1_course_baseline.csv")
        baseline = pd.read_csv(self.out / "course_baseline_test_risks.csv")
        self.assertEqual(len(one_run), len(baseline))
        self.assertEqual(int(one_run["selected"].sum()), int(self.primary["B"].iloc[0]))

    def test_id3_settings_are_fixed_across_checkpoints(self):
        for checkpoint in CHECKPOINTS:
            with open(self.out / f"id3_{checkpoint}_settings.json") as fh:
                settings = json.load(fh)
            self.assertEqual(settings["n_bins"], 3)
            self.assertEqual(settings["max_depth"], 3)
            self.assertEqual(settings["min_samples_split"], 10)
            stats = settings["tree_stats"]
            self.assertGreaterEqual(stats["depth"], 0)
            self.assertLessEqual(stats["depth"], 3)
            self.assertGreater(stats["n_nodes"], 0)
            self.assertGreater(stats["n_leaves"], 0)

    def test_selected_knn_k_is_always_tuned_from_the_grid(self):
        grid = {3, 5, 11, 21, 51}
        for run in ("enrollment", "semester1", "semester2"):
            with open(self.out / f"knn_{run}_settings.json") as fh:
                settings = json.load(fh)
            self.assertIn(settings["k"], grid)
            self.assertGreaterEqual(settings["k"], 3)

    def test_knn_tuning_summary_covers_every_knn_run(self):
        table = pd.read_csv(self.out / "knn_tuning_summary.csv")
        self.assertEqual(len(table), 6)
        self.assertEqual(set(table["run"]),
                         {"P4 kNN enrollment", "P5 kNN semester 1", "P6 kNN semester 2",
                          "A3 kNN raw semester 2", "A4 kNN reduced semester 2",
                          "A6 kNN no Gender enrollment"})
        self.assertTrue(table["k"].isin([3, 5, 11, 21, 51]).all())
        self.assertTrue(table["validation_precision"].between(0, 1).all())

    def test_ablation_table_covers_a1_through_a6(self):
        table = pd.read_csv(self.out / "ablations_A1_A6.csv")
        self.assertEqual(list(table["run"]),
                         ["A1 Course-only baseline", "A2 Random-selection expectation",
                          "A3 kNN raw features (semester 2)", "A4 kNN reduced features (semester 2)",
                          "A5 ID3 no Gender (enrollment)", "A6 kNN no Gender (enrollment)"])
        self.assertTrue(table["precision"].between(0, 1).all())
        self.assertTrue(table["recall"].between(0, 1).all())

    def test_reduced_ablation_actually_drops_the_redundant_columns(self):
        with open(self.out / "knn_reduced_ablation_summary.json") as fh:
            summary = json.load(fh)
        removed = set(summary["removed_features"])
        self.assertEqual(removed, {"International", "Curricular units 1st sem (approved)"})
        self.assertLessEqual(summary["overlap_with_full_selection"], summary["budget"])
        semester2_knn = self.primary[(self.primary["model"] == "kNN (mixed)")
                                     & (self.primary["checkpoint"] == "semester2")].iloc[0]
        self.assertEqual(summary["full"]["reached"], semester2_knn["dropouts_reached"])

    def test_three_distinct_students_are_chosen_for_explanation(self):
        students = json.loads((self.out / "explanation_students.json").read_text())["students"]
        self.assertEqual(len(students), 3)
        self.assertEqual(len(set(students)), 3)

    def test_enrolled_students_are_scored_but_not_relabelled(self):
        with open(self.out / "enrolled_vs_test_risk_summary.json") as fh:
            summary = json.load(fh)
        for run in ("enrolled_id3", "enrolled_knn", "test_id3_P3", "test_knn_P6"):
            self.assertIn(run, summary)
            stats = summary[run]
            for key in ("count", "mean", "min", "max", "median"):
                self.assertIn(key, stats)
            for key in ("mean", "min", "max", "median", "p25", "p75"):
                self.assertGreaterEqual(stats[key], 0)
                self.assertLessEqual(stats[key], 1)


if __name__ == "__main__":
    unittest.main()
