# 6. Evaluation: the advising list

**Goal:** judge each model the way an advising office would use it: how many real dropouts land on a meeting list limited to 10% of students. **Do not use accuracy to choose a winner.**

Code: `select_advising_list`, `capacity_metrics`, `group_metrics`, `CourseBaseline` in `main.py`.

## Build the advising list: `select_advising_list`

For a validation or test partition with `n` students:

1. Budget `B = max(1, floor(0.10 * n))`. That is **72** for both the 726-row validation set and the 726-row test set.
2. Sort students by **descending** dropout risk, then **ascending** row ID for equal scores.
3. Select exactly the first `B`.
4. Return a Boolean mask aligned to the risk Series' index.

Wrap each model's risk array in a pandas Series indexed by the scored DataFrame's row IDs. The tie-break must not use outcomes or sensitive attributes.

Tree leaves and neighbor votes create many ties. For every list, report how many students share the cutoff score and how many of them were selected (`cutoff_tied`, `cutoff_selected`).

## Compute the metrics: `capacity_metrics`

`TP_B` is the number of actual dropouts on the list. `D` is the total number of actual dropouts in the partition.

| Measure | Formula / meaning |
| --- | --- |
| Dropouts reached | `TP_B`: direct answer to the advising question |
| Precision@10% | `TP_B / B`: useful meetings per available slot |
| Recall@10% | `TP_B / D`: fraction of all dropouts reached |
| Missed-dropout rate | `(D - TP_B) / D` |
| Lift@10% | `Precision@10% / (D / n)` |

- Return the keys listed in the docstring: `n`, `budget`, `dropouts`, `reached`, `precision`, `recall`, `missed_rate`, `lift`, `cutoff_score`, `cutoff_tied`, `cutoff_selected`.
- If the partition has no actual dropouts, return `None` for recall, missed rate, and lift.
- Always report `n`, `B`, and `D` alongside the rates.

> **Note: the recall ceiling.** With a fixed budget, the best possible Recall@10% is `min(1, B / D)` when `D > 0`. Even a perfect list cannot reach more than `B` dropouts. Judge recall against this ceiling, not against 100%.

> **Note: prediction is not prevention.** State explicitly in your report that identifying a future dropout does not show that a meeting would prevent the dropout.

## Course-only baseline

`CourseBaseline` is run A1 in the [run matrix](README.md#every-run-you-must-do).

1. `fit`: estimate each course's dropout fraction from the **training** rows, plus the overall training dropout fraction.
2. `predict_risk`: give each student their course's fraction. Give an unseen course the overall training fraction.
3. Evaluate it on the test rows with the **same** budget and tie-break as the models.

## Random-selection expectation

Run A2 in the [run matrix](README.md#every-run-you-must-do). This is a reference value, not a trained model:

- expected dropouts reached = `B * D / n`
- expected precision = `D / n`
- expected recall = `B / n` (the random line in the Recall@10% panel)
- expected lift = 1

It uses the prevalence in the evaluation partition, for context only. It is never used to construct predictions.

## Course outcome description

For description only, compute the full three-label outcome distribution (Graduate / Dropout / Enrolled) for each local course code. Keep these full-data distributions **out of** model fitting.

> **Note: comparing with the paper.** The [dataset paper](https://www.mdpi.com/2306-5729/7/11/146) reports roughly 72% and 70% on-time graduation for Nursing and Social Service, and 8% for Biofuel Production Technologies and for Informatics Engineering. Those percentages include Enrolled students in the denominator, so rates computed on binary labels only will differ. Do not attach course names to codes without a verified local mapping.

## The timeline results

Train each primary model independently at each checkpoint: **three ID3 models and three mixed-distance kNN models**. Evaluate each once on the test rows.

### Six-row test-results table

One row per primary run (P1–P6). Each row must show at least:

| Column | Meaning |
| --- | --- |
| Checkpoint | Enrollment / semester 1 / semester 2 |
| Model | ID3 or kNN (mixed) |
| `n` | Test partition size |
| `B` | Budget |
| `D` | Actual dropouts in the test partition |
| Dropouts reached | `TP_B` |
| Precision@10% | |
| Recall@10% | |
| Lift | |
| Chosen settings | ID3 fixed settings, or the tuned `k` |
| Cutoff tie counts | Students tied at the cutoff score, and how many were selected |

Put the Course baseline and the ablations (A1–A6) in **separate, adjacent** tables, so the six primary rows stay easy to compare.

### Timeline plot

A two-panel figure:

- **Left panel:** Precision@10%. **Right panel:** Recall@10%.
- x-axis: the three checkpoints. One line for ID3 and one for kNN.
- Two horizontal reference lines in each panel: the Course-only baseline and the random-selection expectation.

### Interpretation

- [x] Report the change between checkpoints in **percentage points** for each model.
- [x] Semester results may give a large improvement. Investigate this rather than forcing the pattern.
- [x] Discuss the predictive value of approved units, grades, `Course`, and fee status, and weigh it against the cost of waiting to intervene.

## Save / report

See the [doc 6 items in the evidence checklist](10-submission.md#evaluation-doc-6).

## Check yourself

- Validation budget = 72 and test budget = 72.
- Equal risks are selected in row-ID order.
- All risks lie in `[0, 1]`.
- `python -m unittest discover -s tests -v`

## Discuss in report

Feeds [question 1](09-report.md#q1-earlier-help-or-later-accuracy) (earlier help or later accuracy).

Next: [Explanations and fairness](07-explanations-and-fairness.md).
