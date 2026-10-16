# 8. Data detective and Enrolled students

**Goal:** investigate three quirks of the data, and look carefully at the students whose outcomes are unresolved.

For every mystery, answer with **evidence from the CSV**, and keep evidence clearly separate from hypotheses.

## Mystery 1: the hidden year

**Compute**

- [x] The distinct `(Unemployment rate, Inflation rate, GDP)` triples. The supplied CSV has **10**.
- [x] The size and label distribution of each triple.

**Discuss**

- How might the triples act as a proxy for enrollment cohorts? There is no year column, and ten triples do not prove a one-to-one mapping to years.
- Why might a rule based on GDP fail next year?
- Propose a future-year validation design that uses verified dates.

**Optional:** evaluate on held-out whole economic triples. Call this a *group generalization check*, not a verified chronological split.

## Mystery 2: zero grades

**Compute**

- [x] Zero counts and grade summaries for both semesters, over **all 4,424 rows** of the CSV. Expected values:

  | Semester | Zero grades | Mean grade | Median grade |
  | --- | ---: | ---: | ---: |
  | 1 | 718 | about 10.641 | about 12.286 |
  | 2 | 870 | about 10.230 | 12.2 |

- [x] Cross-tabs of zero grades against enrolled, approved, and without-evaluation units, for each semester.

**Discuss**

- Does the evidence point to failing grades, to no assessment, or to several mechanisms? Do not claim zeros are missing values without evidence.
- What do zeros do to kNN distances and to the ID3 bins?
- Propose a sensitivity comparison: keeping zeros vs an explicit no-assessment indicator with a justified grade treatment, fitted on training data only. (You propose it; running it is not required.)

## Mystery 3: what is dropout?

**Discuss**

- The source counts transfers to another course or institution as dropout. How does that change the advising goal?
- Could a successful transfer be wrongly treated as a bad outcome?
- What extra labels or event dates would distinguish transfer support, delayed completion, and leaving education?

## Enrolled students

1. Score all 794 Enrolled students with the **frozen semester 2 models** (P3 and P6).
2. Produce exploratory risk-score summaries (for example, distribution statistics or histograms).
3. Compare these distributions with the risk scores of the labeled **test** students.
4. State the assumption and the limitation of applying models learned only from resolved outcomes to this group.

**Do not** compute dropout precision, recall, or accuracy for Enrolled students, and **do not** relabel them as Graduate.

## Save / report

See the [doc 8 items in the evidence checklist](10-submission.md#data-detective-doc-8).

## Discuss in report

Feeds [question 6](09-report.md#q6-hidden-year), [question 7](09-report.md#q7-zero-grades-and-label-definition), and [question 8](09-report.md#q8-unresolved-enrolled-students).

Next: [Report and discussion questions](09-report.md).
