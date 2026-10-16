# 10. Validate and submit

**Goal:** confirm that everything runs from scratch, then hand in code, outputs, and the report.

## Final validation

1. Complete every `TODO`. No `raise NotImplementedError` may remain.
2. Run the full test suite. It must pass:

   ```bash
   python -m unittest discover -s tests -v
   ```

   The supplied suite has **46** test methods. They include small, meaningful algorithm checks:
   - Entropy is 0 for a pure node and 1 for an equally mixed binary node. A hand-computed information-gain example selects the expected feature.
   - A tiny ID3 example predicts the expected leaf fractions and handles an unseen category safely.
   - A hand-computed kNN example verifies distances, neighbor order, votes, and ties. Renumbering category codes arbitrarily does not change the categorical mismatches.
   - A ranking example checks the exact budget, the tied cutoff, and the group missed-dropout denominator. All risks lie in `[0, 1]`. The feature snapshots are nested and exclude future columns.

   The suite does **not** run the fitting/tuning helpers or the full `run_experiments` pipeline, so passing it is not enough on its own.
3. Delete your output directory and run the required entry point from a clean state:

   ```bash
   python main.py --data dataset.csv --output results --seed 42
   ```

4. Check that the saved split IDs, predictions, selections, and report tables agree with each other and with the numbers in your PDF.
5. Confirm that ID3 used the fixed settings, and that every kNN choice was made on validation data **before** you looked at the final test results.

## What to submit

1. Completed `main.py`, `decision_tree.py`, and `knn.py`, plus any additional helpers. The required entry point is `python main.py --data dataset.csv --output results --seed 42`; it must run the experiments and save the requested artifacts.
2. Saved split row IDs, fixed ID3 settings, and selected kNN hyperparameters. State the tie-breaking and unseen-value behavior.
3. PDF file (with previous exercises) with results table/plot, Course baseline, distance and redundancy comparisons, three paired explanations, fairness tables and Gender-removal comparison, detective answers, Enrolled summaries, and **clearly labeled answers to all eight required discussion questions**. See the [report checklist](09-report.md#report-contents-checklist).
4. Machine-readable results (CSV or JSON), generated figures, the supplied tests, and any additional checks. Put the instructions for running those extra checks in the PDF. Keep `requirements.txt`, and record the installed package versions in the PDF. Add dependencies only if you can justify them. You do not need to write a README.

Use filenames and experiment labels that tell apart the six timeline models and the additional experiments. You choose the filenames. Your submission must reproduce every reported metric.

## How to hand in

1. Put everything in **one zip file**: your assignment folder with the completed code, `tests/` (including any tests you added), `requirements.txt`, `dataset.csv`, the full output folder (`results/`), and the PDF report.
2. Leave out virtual environments (`.venv/`) and `__pycache__/` folders.
3. Unzip it into an empty folder and check that `python main.py --data dataset.csv --output results --seed 42` still runs there.
4. **Upload the zip to Canvas.**

## Submission checklist

**Code**

- [x] `main.py`, `decision_tree.py`, `knn.py` completed, with no remaining `TODO` raises
- [x] Public method signatures and return formats unchanged
- [x] No forbidden library classifiers, neighbor searches, or distance functions ([doc 1](01-setup.md#allowed-and-forbidden-tools))
- [x] `python -m unittest discover -s tests -v` passes all 46 supplied tests, which are unmodified
- [x] Any additional tests or checks are included
- [x] `python main.py --data dataset.csv --output results --seed 42` runs from a clean output directory

**Report (PDF)**

- [x] Every item in the [evidence checklist](#evidence-checklist) below
- [x] Everything in the [report checklist](09-report.md#report-contents-checklist), including all eight discussion questions under clear labels
- [x] Every number traceable to the saved outputs
- [x] How to run any extra checks you added
- [x] Installed package versions

**Hand-in**

- [x] One zip with code, tests, `requirements.txt`, `dataset.csv`, `results/`, and the PDF, and no `.venv/` or `__pycache__/`
- [ ] Zip uploaded to Canvas

## Evidence checklist

This is the one list of everything your code must save and your report must show. Save each item as machine-readable output (CSV or JSON) or as a figure, **and** present it in the PDF. Items marked *(PDF only)* are written explanations. You choose the filenames, but they must make clear which run each file belongs to.

### Data and splits (doc 3)

- [x] Train, validation, and test row IDs
- [x] Dataset summary: resolved vs Enrolled counts, and any anomalies you found
- [x] Checkpoint timeline figure (drawn by hand or generated by code), with the availability assumption and caveats
- [x] How excluding Enrolled students changes the population your metrics describe *(PDF only)*

### ID3 (doc 4)

- [x] Fixed configuration and validation Precision@10% at each checkpoint
- [x] Fitted tree depth, node count, and leaf count at each checkpoint
- [x] At least one decision path or small tree example
- [x] Tie-breaking and unseen-value behavior *(PDF only)*

### kNN (doc 5)

- [x] Selected `k` and validation Precision@10% for every kNN run (P4–P6, A3, A4, A6)
- [x] Raw vs mixed at semester 2: chosen `k`, test Precision@10%, and test Recall@10% for each
- [x] One student's neighbor list under raw vs mixed distance
- [x] Redundancy evidence: the two correlations and a `Nacionality` × `International` cross-tab
- [x] Reduced-feature run: settings and capacity metrics next to P6
- [x] Tie-breaking and unseen-category behavior *(PDF only)*

### Evaluation (doc 6)

- [x] Test risk scores and selections for every trained run in the [run matrix](README.md#every-run-you-must-do) (all except A2, which is a formula)
- [x] Six-row timeline results table
- [x] Baseline table (Course-only and random expectation) and ablation table, next to it
- [x] Two-panel timeline plot
- [x] Course outcome distributions (descriptive)

### Explanations and fairness (doc 7)

- [x] The three chosen row IDs
- [x] Three side-by-side ID3/kNN explanations
- [x] Group tables for `Gender`, `Scholarship holder`, and `Tuition fees up to date`, for both enrollment models
- [x] Missed-rate difference for each attribute
- [x] Gender-removal comparison: overall metrics and the gender gap, before and after

### Data detective (doc 8)

- [x] Economic-triple table: triple, size, and label distribution
- [x] Zero-grade counts, summaries, and cross-tabs for both semesters
- [x] Answers to all three mysteries, with evidence kept separate from hypotheses *(PDF only)*
- [x] Enrolled risk-score summaries for both semester 2 models, next to the labeled test distribution

## Grading rubric

| Component | Points |
| --- | ---: |
| Correct ID3 implementation, risk scores, and categorical handling | 5 |
| Correct kNN implementation, risk scores, and distance design | 5 |
| Timeline, leakage prevention, shared splits, and tuning protocol | 4 |
| Capacity metrics, Course baseline, results and plots | 4 |
| Paired explanations and interpretation | 3 |
| Fairness audit and Gender-removal experiment | 2 |
| Data detective work, Enrolled analysis, and reproducibility | 2 |
| **Total** | **25** |

Extensions are optional and do not replace required work. No points depend on exceeding an arbitrary performance threshold.
