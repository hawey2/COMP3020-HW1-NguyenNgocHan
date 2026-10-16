# 7. Explanations and fairness

**Goal:** see what each model's prediction means for an individual student, and check which groups the advising list misses.

## Part A: choose three test students

Choose the students **before** looking at their outcomes, using this reproducible rule on the **semester 2 ID3** (P3) test risks:

| Student | Rule |
| --- | --- |
| 1 | Highest ID3 risk |
| 2 | ID3 risk closest to 0.5 |
| 3 | Lowest ID3 risk |

Pick them in this order (1, then 2, then 3), each time choosing only from students not already picked, so no student is picked twice. Break equal scores by the smaller row ID. Refer to students by their anonymous row ID only.

## Part B: explain each student both ways

Use the semester 2 models (P3 and P6). Show the two explanations side by side for each student.

**ID3 explanation** (from `decision_path`):

- [x] each feature and value on the path the student followed;
- [x] the original numeric **bin intervals**, where the feature was binned (translate them with the preprocessor);
- [x] the training counts `(Graduate, Dropout)` at the node reached;
- [x] the dropout risk.

**kNN explanation** (from `kneighbors`, mixed distance):

- [x] the **five** nearest training students, with their distances and true training outcomes;
- [x] a few feature similarities and differences between the student and those neighbors;
- [x] if the tuned `k` is not 5, show the five-neighbor explanation **separately** from the vote over the actual `k`, and report that full vote and risk as well.

Answer in the report:

- Which explanation could guide an advisor's next conversation?
- Which might a student consider fair?
- Does similarity imply a cause, or a useful intervention?
- Name at least **one actionable question** an advisor could ask and **one factor** that should not be treated as destiny.

## Part C: missed-dropout audit

Use the **enrollment** models (P1 and P4), since an early meeting is where this matters most.

For each model's **global** top-10% test list, compute a table with `group_metrics` for each of these attributes:

| Attribute | Coding |
| --- | --- |
| `Gender` | 1 = male, 0 = female |
| `Scholarship holder` | Binary |
| `Tuition fees up to date` | Binary |

Each group row must show: group size, actual dropout count, selected count, dropouts reached, selection rate, and the missed-dropout rate:

```text
group missed-dropout rate = actual group dropouts outside the global list
                           / all actual group dropouts
```

- If a group has no actual dropouts, `group_metrics` returns `NaN` for its missed rate. Show it as `N/A` in the report.
- Report the denominators. Avoid firm conclusions about small groups.
- For each attribute, compute the **difference** in missed-dropout rate between its two groups as group coded 1 minus group coded 0 (for `Gender`: male minus female), and say which group is missed more.
- Always select from **one global budget**. Do **not** allocate 10% separately inside each group.

## Part D: Gender-removal rerun

These are runs A5 and A6 in the [run matrix](README.md#every-run-you-must-do).

1. Remove `Gender` from the enrollment predictors.
2. Rerun both enrollment models with the same partitions and the same global budget:
   - ID3 keeps the fixed configuration;
   - kNN retunes `k` on the validation rows.
3. Keep `Gender` aside for the audit, and recompute the Gender audit table for the new lists.
4. Compare overall performance **and** the gender gap, before and after removal.

Discuss in the report:

- Other features may still carry the gap. Removing a column is not proof of fairness.
- Would debtor and fee flags lead to financial support or to penalties? How would advisors use them?
- Why can't observed associations establish causes?

## Save / report

See the [doc 7 items in the evidence checklist](10-submission.md#explanations-and-fairness-doc-7).

## Check yourself

- The group tables use the same global selection as the enrollment models' test results.
- The group dropouts outside the list plus the group dropouts reached equal the group's dropouts.

## Discuss in report

Feeds [question 4](09-report.md#q4-useful-and-fair-explanations) (explanations) and [question 5](09-report.md#q5-who-is-missed) (who is missed).

Next: [Data detective and Enrolled students](08-data-detective.md).
