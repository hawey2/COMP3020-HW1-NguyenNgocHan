# 3. Data, checkpoints, and splits

**Goal:** define exactly which students and which columns each model may use, and create **one** split that every run shares.

Code: `prepare_cohorts`, `feature_sets`, `split_row_ids` in `main.py`.

## Define the target

1. Keep only the `Dropout` and `Graduate` rows for the main supervised task. That is 3,630 students.
2. Encode **Dropout = 1** and **Graduate = 0**.
3. Set aside all 794 `Enrolled` rows **before** splitting. These students had not completed their course at the observation horizon, so their eventual outcomes are unknown. They are used only in the [exploratory Enrolled analysis](08-data-detective.md#enrolled-students).

> **Note: what the label means.** The [dataset paper](https://www.mdpi.com/2306-5729/7/11/146) records outcomes at the normal course duration: three years, or four for Nursing. Transfers to another course or institution also count as dropout. The cohort covers enrollment years 2008/09–2018/19. You are predicting the dataset's outcome definition, which is not necessarily leaving higher education.

## Create the three checkpoint feature sets

The sets are **nested**: each one contains everything in the one before it.

| Checkpoint | Predictors allowed | Count |
| --- | --- | ---: |
| Enrollment | All predictors except the 12 semester columns | 22 |
| End of semester 1 | Enrollment predictors plus all six `Curricular units 1st sem (...)` columns | 28 |
| End of semester 2 | Previous set plus all six `Curricular units 2nd sem (...)` columns | 34 |

- Each semester contributes six columns: `credited`, `enrolled`, `evaluations`, `approved`, `grade`, and `without evaluations`.
- Credited and enrolled units belong to their own semester checkpoint in this assignment, even though they might be known earlier.
- **Never** use semester 2 information in an earlier model.

## Draw the timeline

Make a figure showing the three prediction checkpoints and the later label horizon. In your report:

- [x] State the assignment assumption: **non-semester columns are available at enrollment.**
- [x] State the caveat: in reality `Debtor`, `Tuition fees up to date`, and scholarship status can change, so their exact measurement times must be verified before operational use. Annual economic figures may also be published after enrollment. Do not claim the file proves they are available in real time.

> **Note: retrospective cohort.** Students who left before a semester ended may already have their outcome reflected in later academic data. The file has no event dates and no checkpoint eligibility, so the main experiment uses the same retrospective cohort at every checkpoint. In your report, discuss why a live evaluation would instead include only students still eligible for advising at each checkpoint.

## Split once

| Setting | Value |
| --- | --- |
| Rows split | The 3,630 labeled rows only |
| Proportions | Stratified 60% training / 20% validation / 20% test |
| Seed | 42 |
| Method (required) | Use `sklearn.model_selection.train_test_split` twice: reserve 20% for test, then 25% of the remaining 80% for validation. Stratify and use seed 42 at both steps. Any other method gives different row IDs, so your numbers would not be comparable. |
| Expected sizes | 2,178 training, 726 validation, 726 test |

1. `split_row_ids` returns a dict with `train`, `validation`, and `test` arrays of **row IDs** (actual `y.index` values, not positions).
2. Select features and labels with `.loc[ids]` so they stay aligned.
3. **Save the row IDs.** Reuse exactly these partitions for both algorithms, all three checkpoints, and every ablation.

## Leakage rules

| Do | Don't |
| --- | --- |
| Fit bins, scalers, category vocabularies, and feature-selection decisions on **training rows only** | Fit anything on validation, test, or Enrolled rows |
| Make model choices (such as `k`) using **validation** results | Change any choice after seeing test performance |
| Freeze all choices, then evaluate on test **once** | Refit the final model on training + validation. For simplicity, final models stay trained on the original training partition. |
| Keep row IDs out of the feature matrix | Infer chronological order from the CSV row order |
| Keep the Enrolled rows separate | Use Enrolled rows for fitting or tuning |

> **Note: what a random split measures.** A random split measures performance on similar historical students. It does **not** show how a model would do on a future enrollment year.

## Save / report

See the [doc 3 items in the evidence checklist](10-submission.md#data-and-splits-doc-3).

## Check yourself

- 3,630 labeled rows, 794 Enrolled rows
- Feature counts 22 / 28 / 34, nested, with no future columns
- Split sizes 2,178 / 726 / 726
- `python -m unittest discover -s tests -p 'test_main.py' -k TestCohortsAndSplits -v`

## Discuss in report

Feeds [question 1](09-report.md#q1-earlier-help-or-later-accuracy) (which checkpoint to use) and [question 8](09-report.md#q8-unresolved-enrolled-students) (Enrolled students).

Next: [ID3 decision tree](04-id3.md).
