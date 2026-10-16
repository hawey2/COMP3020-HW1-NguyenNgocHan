# 9. Report and discussion questions

**Goal:** write a PDF report built entirely from your saved outputs. It must present the required results and answer all eight discussion questions.

## Report contents checklist

The PDF (submitted with the previous exercises) must contain:

- [x] Every result in the [evidence checklist](10-submission.md#evidence-checklist), organized by topic
- [x] **Clearly labeled answers to all eight required discussion questions** (below)
- [x] Installed package versions, and how to run any extra checks you added

A reasonable length is **6–10 pages**, not counting code and detailed tables. Completeness matters more than length.

## How to answer

- Give each question its own clearly labeled heading (for example, "Q1. Earlier help or later accuracy?").
- For questions about model performance or group differences, give the **result**, its **denominator**, and the **checkpoint**.
- Keep observations separate from hypotheses.
- Every number you report must be reproducible from your submitted code and outputs.

## Required discussion questions

### Q1. Earlier help or later accuracy?

How much do Precision@10% and Recall@10% change from enrollment to semester 1 to semester 2, for each model? Which checkpoint would you recommend for a first advising meeting, and what information becomes available too late to support an earlier meeting? Explain why a high score on this retrospective test does not show that an intervention will prevent dropout.

*Evidence to include:* the six-row table, the changes in percentage points, and the timeline plot ([doc 6](06-evaluation.md#the-timeline-results)).

### Q2. Meaningful splits and distances?

Answer the [`Course <= 8.5` question](04-id3.md#discuss-in-report) from the ID3 doc. Explain why a one-unit difference between category codes is not necessarily small, using a parent-education or `Course` example. Compare the raw and mixed kNN neighbor lists for one student, and say which distance you would use.

*Evidence to include:* a decision path or small tree example; the raw vs mixed neighbor lists and metrics ([doc 5](05-knn.md#raw-vs-mixed-distance-comparison)).

### Q3. Duplicated signals?

Use the approved-unit and `Nacionality`/`International` evidence, plus your reduced-feature kNN result. Did removing the two columns change the advising list (for example, how many of the 72 selected students differ) or its performance? Why can repeated information affect kNN more directly than a single tree split?

*Evidence to include:* the correlations, the cross-tab, and reduced vs full kNN metrics ([doc 5](05-knn.md#reduced-feature-ablation)).

### Q4. Useful and fair explanations?

For the same three test students, which parts of the tree path and the neighbor evidence could guide an advisor's conversation? Which parts might a student dispute? Name one actionable question and one factor that should not be treated as destiny. Similarity and correlation do not establish causes.

*Evidence to include:* the three side-by-side explanations ([doc 7](07-explanations-and-fairness.md#part-b-explain-each-student-both-ways)).

### Q5. Who is missed?

Compare missed-dropout rates for the Gender, scholarship, and fee-status groups under one global 10% budget. After removing `Gender` from the predictors, does the gender gap remain? Explain how related features could carry similar information, and discuss whether debt or fee flags should trigger support or a penalty.

*Evidence to include:* the group tables with denominators, the missed-rate differences, and the before/after Gender-removal comparison ([doc 7](07-explanations-and-fairness.md#part-c-missed-dropout-audit)).

### Q6. Hidden year?

What do the 10 economic-value triples reveal, and what can they not prove without actual enrollment-year labels? Why might a GDP split fail for future students? Describe a feasible future-cohort validation design.

*Evidence to include:* the economic-triple table ([doc 8](08-data-detective.md#mystery-1-the-hidden-year)).

### Q7. Zero grades and label definition?

What does the cross-tab evidence suggest about a grade of 0? How would the alternative treatments affect ID3 bins and kNN distances? What does the source definition of dropout do to students who transfer courses or institutions, and what additional data would help distinguish their needs?

*Evidence to include:* the zero-grade counts and cross-tabs ([doc 8](08-data-detective.md#mystery-2-zero-grades)) and the dropout-definition discussion ([doc 8](08-data-detective.md#mystery-3-what-is-dropout)).

### Q8. Unresolved Enrolled students?

How do their semester 2 risk-score distributions compare with those of the labeled test students? Why can't you compute eventual-dropout performance for this group from the supplied CSV?

*Evidence to include:* the Enrolled vs labeled-test score summaries ([doc 8](08-data-detective.md#enrolled-students)).

Next: [Validate and submit](10-submission.md).
