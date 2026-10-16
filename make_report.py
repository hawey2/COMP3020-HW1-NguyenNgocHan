"""Generate the HW1 PDF report from the saved artifacts in results/.

Run after main.py:  python make_report.py --results results --out report.pdf
Uses only numpy/pandas/matplotlib (already required by the assignment).
"""

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAGE_W, PAGE_H = 8.5, 11
LEFT = 0.082
RIGHT = 0.082
LINE_H = 9.8          # points between body lines
FONT = "DejaVu Sans Mono"
BODY = 8.0
COLS = 112            # character columns that fit between the margins
TOP_Y = 0.945
BOTTOM_Y = 0.055


class Report:
    def __init__(self, pdf):
        self.pdf = pdf
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = TOP_Y

    def _new_fig(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = TOP_Y

    def _tick(self, lines=1):
        self.y -= LINE_H * lines / (PAGE_H * 72.0)
        if self.y < BOTTOM_Y:
            self._new_fig()

    def _line(self, text, size=BODY, color="black", weight="normal", indent=""):
        self.fig.text(LEFT, self.y, indent + text, family=FONT, fontsize=size,
                      color=color, fontweight=weight, va="top", ha="left")
        self._tick()

    def para(self, text, size=BODY, color="black", weight="normal"):
        for chunk in textwrap.wrap(text, width=COLS) or [""]:
            self._line(chunk, size=size, color=color, weight=weight)

    def title(self, text, size=15):
        self.y -= 6 / (PAGE_H * 72.0)
        self._line(text, size=size, weight="bold")
        self._tick(0.6)

    def section(self, text, size=11):
        self.y -= 8 / (PAGE_H * 72.0)
        self._line(text, size=size, weight="bold")
        self._tick(0.6)

    def sub(self, text, size=9):
        self._line(text, size=size, weight="bold")

    def spacer(self, lines=1):
        self.y -= LINE_H * lines / (PAGE_H * 72.0)
        if self.y < BOTTOM_Y:
            self._new_fig()

    def table(self, header, rows, size=7.4, widths=None):
        widths = widths or [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) + 2
                            for i in range(len(header))]
        self._line("".join(str(header[i]).ljust(widths[i]) for i in range(len(header))),
                   size=size, weight="bold")
        self._line("-" * int(sum(widths)), size=size, color="#555555")
        for row in rows:
            self._line("".join(str(row[i]).ljust(widths[i]) for i in range(len(row))), size=size)

    def image(self, path, caption, width=0.82, height=0.40):
        need = height + 0.08
        if self.y - BOTTOM_Y < need:
            self._new_fig()
        ax = self.fig.add_axes([LEFT, self.y - height, width, height])
        ax.imshow(plt.imread(path))
        ax.axis("off")
        self.y -= (height + 0.015)
        for chunk in textwrap.wrap(caption, width=COLS - 4):
            self._line("   " + chunk, size=7.4, color="#333333")
        self._tick(0.5)

    def close(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def pct(x):
    return "None" if x is None else f"{100*x:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report.pdf"))
    args = parser.parse_args()
    r = args.results

    timeline = pd.read_csv(r / "timeline_results.csv")
    baseline = pd.read_csv(r / "baseline_and_random.csv")
    fair = pd.read_csv(r / "fairness_group_tables.csv")
    gender_csv = pd.read_csv(r / "gender_removal_metrics.csv")
    raw_mix = pd.read_csv(r / "knn_raw_vs_mixed.csv")
    reduced = pd.read_csv(r / "knn_reduced_ablation.csv")
    zero = pd.read_csv(r / "zero_grade_summary.csv")
    econ = pd.read_csv(r / "economic_triples.csv")
    import json
    enr = json.load(open(r / "enrolled_vs_test_risk_summary.json"))
    expl = json.load(open(r / "explanations.json"))
    expl_ids = json.load(open(r / "explanation_students.json"))["students"]
    red_overlap = json.load(open(r / "knn_reduced_ablation_summary.json"))
    redund = json.load(open(r / "redundancy_evidence.json"))
    gen_rm = json.load(open(r / "gender_removal_settings.json"))
    abl = pd.read_csv(r / "ablations_A1_A6.csv")
    courses = pd.read_csv(r / "course_distributions.csv")
    knn_tuning = pd.read_csv(r / "knn_tuning_summary.csv")
    crosstab = pd.read_csv(r / "nacionality_international_crosstab.csv", index_col=0)

    def row(cp, model):
        return timeline[(timeline.checkpoint == cp) & (timeline.model == model)].iloc[0]

    def att(model, attr, g):
        t = fair[(fair.model == model) & (fair.attribute == attr) & (fair.group == g)]
        return t.iloc[0]

    b = baseline[baseline.reference.str.startswith("Course")].iloc[0]
    rand = baseline[baseline.reference.str.startswith("Random")].iloc[0]

    with PdfPages(args.out) as pdf:
        R = Report(pdf)

        R.title("Early Dropout Prediction with a From-Scratch ID3 Tree and kNN")
        R._line("An advising-list experiment on a Portuguese higher-education cohort", size=9.5, color="#333333")
        R.spacer(1)
        R.section("1.  Problem and protocol")
        R.para(
            "An advising office can meet only a fraction of a cohort, so the deployment task is a "
            "constrained ranking: from each evaluation partition we may place B = floor(0.10 * n) students "
            "on a meeting list and we judge each model by how many real dropouts that list captures "
            "(Precision@10% = reached/B, useful meetings per slot; Recall@10% = reached/D, fraction of all "
            "dropouts reached). Accuracy is deliberately unused for model choice because a 61%-dropout "
            "test set makes a trivially accurate model useless for triage. We implement categorical "
            "multiway ID3 and kNN from scratch and make the list at three nested checkpoints - enrollment "
            "(22 features), end of semester 1 (28) and end of semester 2 (34) - trading earlier help "
            "against later signal. Every run draws boolean risk scores, ranks by descending risk with "
            "ascending row ID as the tie-break, and selects exactly B. All ties fall to 0 (Graduate) for "
            "class prediction."
        )
        R.para(
            "The file holds 4,424 records with 34 predictors and no missing cells or duplicate rows. We "
            "keep the 3,630 resolved outcomes (Dropout = 1, 1,421 students; Graduate = 0, 2,209 students; "
            "binary prevalence 39.1%) and set aside all 794 Enrolled students before any fitting. That "
            "exclusion changes the population every metric below describes: precision, recall and lift are "
            "computed only over students whose outcome is already resolved, so the denominator is the "
            "resolved subcohort and its 39.1% dropout prevalence is higher than on the full intake "
            "(1,421/4,424 = 32.1%). The numbers therefore characterise 'students with a known outcome', "
            "not a full enrollment year, and care is needed when quoting them for planning."
        )
        R.spacer(0.3)
        R.image(r / "checkpoint_timeline.png",
                "Figure 1. Feature availability and the prediction horizon at the three checkpoints. "
                "The 6+6 curricular columns are the only time-gated features; debt, fee status, "
                "scholarship, prior qualification and the economic indicators are treated as available at "
                "enrollment.", width=0.80, height=0.28)
        R.para(
            "One stratified split is shared by every run: train_test_split twice with seed 42 - 20% test, "
            "then 25% of the remainder for validation - giving 2,178 / 726 / 726 rows (saved in "
            "split_ids.json). Bin edges, scaling statistics, and k are learned on training rows only; all "
            "choices are frozen before a single test evaluation."
        )
        R.para(
            "Data checks (doc 1): `--inspect` reproduces the expected facts - 4,424 records, 4,424 "
            "distinct row positions, 34 predictors, 2,209 / 1,421 / 794 Graduate / Dropout / Enrolled, "
            "zero empty cells and zero fully duplicated rows, so no row was deleted or imputed. Column "
            "types, observed value ranges and cardinalities are saved in data_checks.csv and the value "
            "frequencies of all 17 nominal columns in categorical_frequencies.csv. Two coding "
            "observations are flagged rather than corrected: Course is stored as codes 1-17 with no "
            "verified codebook, and the Nacionality x International cross-tab is fully deterministic "
            "(Section 4), so both are kept nominal and never given an invented numeric order."
        )
        R.para(
            "Availability assumption and caveats (doc 3): we assume that all non-semester columns were "
            "already recorded at enrollment and only later appear to be 'current' at the semester "
            "checkpoints; the file supplies no recording timestamps, so this is a modelling assumption, "
            "not a verified chronology, and it makes the enrollment model partly retrospective. In reality "
            "Debtor, Tuition fees up to date, and Scholarship holder can all change during the year, so "
            "their exact measurement times must be verified before any operational use; here they are "
            "frozen as enrollment snapshots. Nothing is dropped as unavailable at a checkpoint - features "
            "are nested by definition, so a later model strictly sees a superset of the earlier one "
            "(22 / 28 / 34 columns, fixed original order). Consequently the timeline measures how much "
            "later information helps, not a leak-free live deployment: a real deployment would also need "
            "the student to still be enrolled at that date, which we cannot check without event dates."
        )
        R.section("2.  Models and settings")
        R.sub("ID3 (fixed configuration, no search)")
        R.para(
            "Numeric columns are binned into up to 3 quantile bins whose internal boundaries are learned "
            "on training rows and reused unchanged (a value on a boundary enters the higher bin; outer "
            "bins extend to infinity). Splits are multiway over nominal categories - no numeric threshold "
            "is ever placed on a code - and gain ties resolve to the first column in original order. "
            "Settings for every ID3 run: n_bins = 3, max_depth = 3, min_samples_split = 10, gain tolerance "
            "1e-12. Fit quality on validation (Precision@10%) was 0.847 (enrollment), 0.944 (semester 1) "
            "and 0.903 (semester 2); the fitted trees all reached depth 3 with 241 / 151 / 259 nodes and "
            "217 / 133 / 229 leaves respectively. One ordinal assumption is load-bearing: Application "
            "order is treated as an ordinary numeric feature and quantile-binned like any other, which is "
            "defensible only if the stored value is a genuine order-of-application rank. The file has no "
            "application-date field to confirm that, so we keep it ordinal but flag it as an assumption "
            "rather than a verified interpretation."
        )
        R.sub("kNN (mixed distance, tuned k)")
        R.para(
            "The metric is d(x,z)=sqrt(sum_j ((x_j - z_j)/train_std_j)^2 over numeric j + sum over nominal j "
            "of 1[x_j != z_j]): numeric features are standardized by training means and population "
            "standard deviations (constant columns use scale 1), while every categorical mismatch "
            "contributes exactly 1 regardless of code magnitude. Voting is uniform over the k nearest "
            "training rows and risk = dropouts/k; ties in class favor 0 and ties in distance favor the "
            "smaller row ID (never stored as a feature). k is chosen from {3,5,11,21,51} per checkpoint by "
            "validation Precision@10% with larger k preferred on ties, yielding k = 21 (enrollment, val "
            "0.903), k = 51 (semester 1, val 1.00) and k = 51 (semester 2, val 1.00)."
        )
        R.spacer(0.4)
        R.sub("kNN tuning summary (selected k and validation Precision@10% for every kNN run)")
        R.table(
            ["run", "distance", "selected k", "val prec@10%"],
            [[t.run, t.distance, int(t.k), f"{t.validation_precision:.3f}"]
             for _, t in knn_tuning.iterrows()],
        )
        R.section("3.  Timeline results")
        R.para("Test partition n = 726, budget B = 72, actual dropouts D = 284 for every row.")
        R.table(
            ["checkpoint", "model", "reached", "prec@10%", "rec@10%", "lift", "settings", "cut_tie/sel"],
            [[
                row(cp, m).checkpoint, row(cp, m).model, int(row(cp, m).dropouts_reached),
                f"{row(cp, m).precision:.3f}", f"{row(cp, m).recall:.3f}", f"{row(cp, m).lift:.2f}",
                (f"k={row(cp,m).settings.split('k=')[1].split(',')[0]}" if m.startswith('kNN') else "fixed"),
                f"{int(row(cp,m).cutoff_tied)}/{int(row(cp,m).cutoff_selected)}",
            ] for cp in ["enrollment", "semester1", "semester2"] for m in ["ID3", "kNN (mixed)"]],
        )
        R.spacer(0.5)
        R.image(r / "timeline_plot.png", "Figure 2. Precision@10% (left) and Recall@10% (right) across "
                "checkpoints; dashed line = Course-only baseline, dotted = random expectation.")
        R.para(
            "Both models improve markedly from enrollment to the second-semester checkpoint, but the "
            "trajectory is not monotone. For ID3, Precision@10% rises 11.1 points from enrollment (0.847) "
            "to semester 1 (0.958) then eases to 0.972 at semester 2, while Recall@10% climbs 2.8 then 0.4 "
            "points to 0.246. kNN jumps from 0.833 to a perfect 1.000 at semester 1 and stays there, its "
            "Recall@10% rising to 0.254. The large semester-1 step is driven by curricular signal: first-"
            "semester approved units and grades begin to separate dropouts, which is why ID3's first split "
            "is 'Curricular units 2nd sem (approved)' at the later checkpoint. The 72-slot budget caps "
            "Recall@10% at min(1, 72/284) = 0.254, so the kNN lists are already at that ceiling - the "
            "binding constraint at later checkpoints is the budget, not the model. Nothing improves by "
            "waiting for semester 2 beyond semester 1."
        )
        R.table(
            ["reference", "reached", "prec@10%", "rec@10%", "lift"],
            [["Course-only baseline (A1)", int(b.reached), f"{b.precision:.3f}", f"{b.recall:.3f}", f"{b.lift:.2f}"],
             ["Random expectation (A2)", f"{rand.reached:.1f}", f"{rand.precision:.3f}", f"{rand.recall:.3f}", "1.00"]],
        )
        R.spacer(0.5)
        R.para(
            "The Course-only baseline reaches 50 of 284 dropouts (Precision@10% 0.694, lift 1.78) without "
            "any enrollment or academic feature; random selection sets the floor at Precision@10% 0.391 "
            "(= the dropout prevalence) and Recall@10% 0.099. Both learned models beat the baseline at "
            "every checkpoint, but even the best enrollment list (0.847) is only 2.16x the prevalence, so "
            "a large share of appended slots still contain non-dropouts - a list, not a diagnosis."
        )
        R.spacer(0.5)
        R.sub("Ablations A3-A6 on the same test rows (full A1-A6 table in ablations_A1_A6.csv)")
        R.table(
            ["run", "settings", "reached", "prec@10%", "rec@10%", "lift"],
            [[row_name, abl[abl.run == row_name].iloc[0].chosen_settings,
              int(abl[abl.run == row_name].iloc[0].reached),
              f"{abl[abl.run == row_name].iloc[0].precision:.3f}",
              f"{abl[abl.run == row_name].iloc[0].recall:.3f}",
              f"{abl[abl.run == row_name].iloc[0].lift:.2f}"]
             for row_name in ["A3 kNN raw features (semester 2)",
                              "A4 kNN reduced features (semester 2)",
                              "A5 ID3 no Gender (enrollment)",
                              "A6 kNN no Gender (enrollment)"]],
        )
        R.spacer(0.5)
        R.sub("Course outcome distributions (descriptive; all 4,424 rows, three labels kept out of fitting)")
        R.table(
            ["course", "Graduate", "Dropout", "Enrolled", "binary drop-rate"],
            [[int(c.Course), int(c.Graduate), int(c.Dropout), int(c.Enrolled),
              f"{c.Dropout / max(1, c.Graduate + c.Dropout):.3f}"]
             for _, c in courses.iterrows()],
        )
        R.para(
            "These full three-label counts are descriptive only and never enter fitting; the binary "
            "drop-rate column drops Enrolled from its denominator, so it is not comparable with the "
            "paper's on-time-graduation rates (which include Enrolled). Course 12 alone holds about a "
            "quarter of the cohort and course 1 holds 12 students, so several apparent course contrasts "
            "rest on small denominators."
        )
        R.para(
            "Q1. Earlier help or later accuracy? Moving from enrollment to semester 1 lifts Precision@10% "
            "by about 11 points and Recall@10% by 3-4 points; semester 2 adds almost nothing for either "
            "model. Because accuracy at the earlier gate is already strong (0.83-0.85) and the whole point "
            "is to intervene while it can still help, we recommend a first meeting at enrollment (or as "
            "soon as semester-1 grades post) and treat semester-2 features as confirmation, not as the "
            "entry point. Information that arrives too late to help includes second-semester approved "
            "units and grades - decisive here but only available after much of the risk has crystallized. "
            "A high retrospective score measures agreement with historical labels at the normal "
            "completion horizon; it does not demonstrate that a meeting prevents dropout, because the "
            "experiment contains no intervention, no counterfactual, and no reason-for-leaving field."
        )
        R.section("4.  Distance design and duplicated signals")
        R.para(
            "A Pearson correlation of 0.916 links first- and second-semester approved units, and 0.915 "
            "links the stored Nacionality codes to International - but the second figure is an artifact: "
            "the Nacionality x International cross-tab shows all 3,544 domestic (code 1) students are "
            "International=0 and all 80 foreign students are International=1, a deterministic coding, not "
            "an ordinal relationship. Distance choice is therefore consequential for kNN."
        )
        R.table(
            ["Nacionality", "International=0", "International=1", "total"],
            [["1 (domestic)", int(crosstab.loc[1].get("0", 0)), int(crosstab.loc[1].get("1", 0)),
              int(crosstab.loc[1].sum())],
             ["2-21 (foreign)", int(crosstab.drop(index=1)["0"].sum()),
              int(crosstab.drop(index=1)["1"].sum()), int(crosstab.drop(index=1).to_numpy().sum())]],
        )
        R.spacer(0.5)
        R.table(
            ["semester-2 kNN run", "k", "val prec@10%", "test prec@10%", "test rec@10%", "reached"],
            [["raw (A3)", raw_mix.iloc[0].settings.split("k=")[1].split(",")[0],
              f"{raw_mix.iloc[0].validation_precision:.3f}",
              f"{raw_mix.iloc[0].test_precision:.3f}", f"{raw_mix.iloc[0].test_recall:.3f}",
              int(raw_mix.iloc[0].dropouts_reached)],
             ["mixed (P6)", raw_mix.iloc[1].settings.split("k=")[1].split(",")[0],
              f"{raw_mix.iloc[1].validation_precision:.3f}",
              f"{raw_mix.iloc[1].test_precision:.3f}", f"{raw_mix.iloc[1].test_recall:.3f}",
              int(raw_mix.iloc[1].dropouts_reached)]],
        )
        R.spacer(0.5)
        R.para(
            "For the highest-risk test student (row 2914) the five nearest neighbours are all dropouts "
            "under both metrics, yet the spaces differ in kind: raw distances span 9.71-12.21 in "
            "arbitrary code units (a few large integer codes dominate), whereas the mixed distances span "
            "2.99-3.46 from standardized academics plus categorical mismatches and retrieve a different, "
            "tighter neighbour set. Renumbering the category codes leaves the mixed distances and "
            "neighbour order identical but scrambles the raw ones - which is why we would deploy the "
            "mixed metric, where each nominal variable contributes a bounded 0/1 penalty and each numeric "
            "variable at most its scaled deviation. An unseen category counts as a mismatch against every "
            "differing training category; neither distance invents a numeric order among codes."
        )
        R.para(
            "Removing both International and 'Curricular units 1st sem (approved)' at semester 2 "
            "(A4, k retuned to k=51) leaves Precision@10% at 1.000 and Recall@10% at 0.254, and shares "
            f"{red_overlap['overlap_with_full_selection']} of 72 selected students with the full model. "
            "Redundancy has almost no cost here because both features carry little unique information once "
            "the second-semester approved units are present; the ablation confirms rather than overturns "
            "the full model. Q2 and Q3: a threshold split such as 'Course <= 8.5' would group courses "
            "8,9,...,17 against 1,...,7 with no evidence that the two halves differ - codes are labels, "
            "so one unit of distance between parent-education codes 24 and 25 means nothing. Our "
            "multiway split instead branches per observed course value, and an unseen course simply stops "
            "at the current node and inherits its training risk. Binning does lose resolution: a "
            "continuous grade becomes one of three ordered labels, so grade 6 and grade 12 are equal in "
            "the tree though not to kNN. Correlated features add to kNN's distance twice (approved units "
            "enter via both semesters), which is why a tree that can pick just one representative split "
            "is less exposed to double-counting - though it too can split on one redundant copy and hide "
            "the rest."
        )
        R.close()
        R = Report(pdf)
        R.section("5.  Individual explanations (semester-2 models P3 and P6)")
        R.para(
            "Three test students were drawn before reading their outcomes by the semester-2 ID3 risk: "
            "highest, closest to 0.5, and lowest. Both frozen models explain each, side by side."
        )
        for sid, truth in zip(expl_ids, [1, 0, 0]):
            e = next(x for x in expl if x["student_row_id"] == sid)
            R.sub(f"Student row_id {sid}  (true outcome: {'Dropout' if truth else 'Graduate'})")
            path_txt = " -> ".join(
                f"{s['feature']}={s.get('display_value')}" if s['feature'] else "leaf"
                for s in e["id3"]["path"]
            )
            last = e["id3"]["path"][-1]
            R.para(
                f"ID3 risk {e['id3']['risk']:.3f}. Path (numeric features shown as the original bin "
                f"interval): {path_txt}. The reached leaf holds "
                f"{last['counts_graduate']} graduates and {last['counts_dropout']} dropouts "
                f"(training dropout rate {last['risk']:.3f})."
            )
            nb = ", ".join(f"{n['row_id']}(d={n['distance']:.2f},{'D' if n['outcome'] else 'G'})"
                            for n in e["knn"]["five_neighbors"])
            R.para(
                f"kNN risk {e['knn']['risk_over_tuned_k']:.3f} (tuned k={e['knn']['tuned_k']}). Five nearest "
                f"training students (row_id, distance, outcome): {nb}. The full-{e['knn']['tuned_k']} vote "
                f"gives the same {e['knn']['risk_over_tuned_k']:.3f}."
            )
            comparison_rows = []
            for c, n in zip(e["knn"]["comparisons"], e["knn"]["five_neighbors"]):
                cats = ", ".join(c["categorical_mismatches"][:3]) or "none"
                gaps = ", ".join(f"{d['feature']}: {d['student']:.0f} vs {d['neighbor']:.0f}"
                                 for d in c["largest_numeric_differences"][:2]) or "none"
                comparison_rows.append([n["row_id"], f"{n['distance']:.2f}",
                                        "D" if n["outcome"] else "G", cats, gaps])
            R.table(["neighbour", "dist", "out", "category mismatches", "largest numeric gaps"],
                    comparison_rows, size=7.0)
            R.spacer(0.4)
        R.para(
            "Q4. Useful and fair explanations. For row 39 the tree isolates a very pure leaf "
            "(Curricular units 2nd sem (approved) in the lowest bin, then Course = 8): every training "
            "student there dropped out, so ID3 reports risk 1.000, while the 51-neighbour vote is a "
            "calmer 0.588 - the two models disagree, and a conversation would be best grounded in the "
            "kNN evidence plus the concrete fact 'no units approved'. The advisor's actionable question is "
            "'Which specific units did you fail or withdraw from this semester, and what got in the way?' "
            "A factor that is not destiny is Course: it is a legitimate predictor but a proxy for cohort "
            "curriculum, not a verdict on a student. Row 26 (risk 0.000 ID3, 0.098 kNN, true Graduate) and "
            "row 336 (ID3 leaf 1/1 = 0.500, kNN 0.098, true Graduate) show the tree's granularity limit - "
            "a 50/50 leaf is thin data, not neutrality - and a student may fairly dispute that their "
            "father's qualification code places them next to a near-zero-risk leaf. Similarity and "
            "correlation are not causation: neighbours share a label, not a reason, and a meeting is only "
            "as useful as what it changes."
        )
        R.section("6.  Who is missed (enrollment lists, one global 10% budget)")
        for attr in ["Gender", "Scholarship holder", "Tuition fees up to date"]:
            R.sub(attr)
            rws = []
            for model in ["P1_id3", "P4_knn"]:
                for g in sorted([0, 1]):
                    t = att(model, attr, g)
                    rws.append([model, g, int(t.n), int(t.dropouts), int(t.selected),
                                int(t.reached), f"{t.selection_rate:.3f}", f"{t.missed_rate:.3f}"])
            R.table(["model", "group", "n", "dropouts", "selected", "reached", "sel_rate", "missed_rate"],
                    rws)
            d1 = att("P1_id3", attr, 1).missed_rate - att("P1_id3", attr, 0).missed_rate
            R.para(f"Missed-dropout-rate difference (group 1 minus group 0): P1 ID3 {d1:+.3f}; "
                   f"P4 kNN {att('P4_knn',attr,1).missed_rate - att('P4_knn',attr,0).missed_rate:+.3f}.")
        R.spacer(0.4)
        R.para(
            "Q5. Under the enrollment lists, female students (Gender 0, 130 dropouts) have the higher "
            "missed-dropout rate than male students in both P1 (0.815 vs 0.760, gap -0.056) and P4 (0.838 "
            "vs 0.747, gap -0.092): women who drop out are slightly less likely to appear on the list. "
            "The largest disparity is not gender but fee status: students not up to date on fees "
            "(Tuition fees up to date = 0, 97 dropouts) are flagged almost everywhere and are missed far "
            "less (0.495 for P1) than up-to-date students (0.936), a +0.441 gap - the model leans on the "
            "debt/fee flag as a strong risk cue. Scholarship holders show a smaller gap (P1 +0.067), again "
            "favouring non-holders. "
        )
        R.para(
            "Gender-removal rerun (A5, A6): ID3 performance and its gender gap are unchanged "
            "(Precision@10% 0.847, gap -0.056), so related features already carry that signal; kNN "
            f"improves slightly (Precision@10% 0.833 to 0.875, its tuned k retuned to {gen_rm['A6_knn']['k']}) "
            "and its gender gap narrows from -0.092 to "
            "-0.040 without vanishing. This is the central caution: dropping a column is not proof of "
            "fairness, because proxies (displacement, debt, fee flags, parental qualification) can "
            "re-encode it. Debtor and fee flags are double-edged - they can route a student to "
            "bursary/fee support, or they can quietly lower the priority given to an unflagged struggling "
            "student; advisors must use them as outreach prompts, never as a penalty. Finally, none of "
            "these associations establish cause: selection into late fee payment or scholarship status is "
            "itself entangled with prior circumstances."
        )
        R.section("7.  Data detective")
        R.table(
            ["unemp", "infl", "GDP", "n", "dropouts", "grad", "rate"],
            [[f"{t['Unemployment rate']}", f"{t['Inflation rate']}", f"{t['GDP']}", int(t.n),
              int(t.dropouts), int(t.graduates), f"{t.dropout_rate:.3f}"]
             for _, t in econ.iterrows()],
        )
        R.spacer(0.4)
        R.para(
            "Q6. Evidence from the CSV: there are exactly 10 distinct (Unemployment, Inflation, GDP) "
            "triples over the 3,630 labeled rows, spanning 291-452 students each, with dropout rates from "
            "0.301 to 0.447. Hypothesis (not proven by this file): their repeated, block-like values could "
            "be yearly macro contexts acting as a cohort proxy, but there is no year column and 10 triples "
            "cannot prove a one-to-one year mapping. Why a GDP split may fail for future students: next "
            "year's economy is a new triple the model has never seen, and the association is at the cohort "
            "level, not the individual level. Feasible future-cohort validation design: hold out whole "
            "economic triples (train on some triples, test on the rest) as an explicit group-generalization "
            "check, and, better, refit on later-labeled years once true enrollment dates are supplied."
        )
        R.table(
            ["sem", "grade col", "zeros", "mean", "median", "nonzero mean"],
            [[z.semester, str(z.column).replace("Curricular units ", ""), int(z.zero_count),
              f"{z.mean_grade:.3f}", f"{z.median_grade:.3f}", f"{z.nonzero_mean:.3f}"]
             for _, z in zero.iterrows()],
        )
        R.spacer(0.4)
        R.image(r / "enrolled_risk_histogram.png",
                "Figure 3. Sem-2 ID3 (P3) risk for the 794 Enrolled students (left) vs the 726 labeled "
                "test students (right).", width=0.86, height=0.34)
        R.para(
            "Q7. Evidence from the CSV: over all 4,424 rows, semester-1 grades are 0 for 718 students and "
            "semester-2 for 870; zeros concentrate almost entirely in approved = 0 and in enrolled = 0 "
            "(sem 1: all 718 zeros sit at approved = 0; 180 at enrolled = 0), while a separate tail of "
            "zeros co-occurs with enrolled units and continued enrolment. Hypothesis: a zero is a stored "
            "state, not a missing cell and not simply a failing grade - the cross-tabs point to several "
            "mechanisms (non-assessment, non-participation, and true low performance) rather than one, and "
            "we do not claim zeros are missing without evidence. Effect on the models: a zero sits in the "
            "lowest ID3 bin together with genuinely low grades, blurring 'no assessment' with 'low "
            "performance' in the tree, and it dominates the standardized numeric distance that kNN computes "
            "for grades. Proposed sensitivity comparison (not run here): keep zeros versus an explicit "
            "no-assessment indicator with a justified grade treatment, fitted on training data only. Label "
            "definition: the source counts transfers to another course or institution as dropout, so a "
            "successful transfer is counted as a bad outcome and the model predicts the label, not an unmet "
            "need; distinguishing transfer support, delayed completion, and true leaving would require an "
            "outcome/transfer code and event dates."
        )
        R.para(
            "Q8. The Enrolled students' semester-2 ID3 risks center lower than the resolved test set "
            f"(mean {enr['enrolled_id3']['mean']:.3f}, median {enr['enrolled_id3']['median']:.3f} vs test "
            f"{enr['test_id3_P3']['mean']:.3f} / {enr['test_id3_P3']['median']:.3f}); kNN shows the same "
            f"direction (Enrolled median {enr['enrolled_knn']['median']:.3f} vs test "
            f"{enr['test_knn_P6']['median']:.3f}). This is descriptive only. We cannot compute eventual-"
            "dropout precision/recall/accuracy for them because their outcomes are unresolved in the file "
            "- absence of a resolution is not a Graduate. Applying a model trained on resolved students "
            "assumes the Enrolled share the same feature-outcome relationships; they are by definition "
            "still in progress and may have a different risk structure, so the scores are exploratory "
            "triaging input, not measured performance."
        )
        R.section("8.  Limitations, reproducibility, and environment")
        R.para(
            "Caveats. (i) The enrollment model reads a partly retrospective snapshot of debt, fee status, "
            "scholarship and economic indicators, whose recording times are unverified (Section 1). "
            "(ii) The cohort is retrospective with "
            "no event dates or checkpoint eligibility, so a live evaluation would include only students "
            "still eligible at each checkpoint. (iii) A random split measures generalization to similar "
            "historical students, not to a future enrollment year. (iv) Risk scores are empirical label "
            "frequencies (tree leaves, neighbour votes), not calibrated probabilities, and identification "
            "is not prevention. (v) Local course and education codes are unverified labels; we keep them "
            "nominal and attach no external codebook."
        )
        R.para(
            "Tie-breaking and unseen values. Advising ties fall to the smaller row ID (never to a label or "
            "sensitive attribute). In ID3 an unseen category stops traversal at the current node and "
            "returns its training risk; in kNN an unseen code counts as a mismatch against every "
            "differing training category. Bins are learned on training rows and reused unchanged; scaling "
            "uses training means and population standard deviations with constant features at scale 1."
        )
        R.para(
            "Reproducibility. Everything is regenerated by 'python main.py --data dataset.csv --output "
            "results --seed 42' after deleting results/; the row IDs, settings, selections, and metrics "
            "above are the saved artifacts (CSV/JSON) plus figures timeline_plot.png, "
            "checkpoint_timeline.png and enrolled_risk_histogram.png. Selection masks for all eleven "
            "trained runs are exported as selection_<run>.csv and consolidated in "
            "test_selections_all_runs.csv, and ablations_A1_A6.csv holds the baseline plus all six "
            "ablation runs side by side. Checks: 'python -m unittest discover -s tests -v' runs 66 tests "
            "- the 46 supplied tests, unmodified, plus 20 added checks in "
            "tests/test_student_extras.py (bin-interval translation, budget and tie-break invariants, "
            "feature-set nesting, baseline fallback, and an end-to-end pipeline run on a small synthetic "
            "file verifying every required artifact). This report itself is produced "
            "by 'python make_report.py --results results --out report.pdf'. Environment: Python 3.12.2, "
            "numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.7 (versions recorded at "
            "run time). scikit-learn is used only for train_test_split; both classifiers, distances, "
            "binning, gain, votes and metrics are implemented from scratch."
        )
        R.close()


if __name__ == "__main__":
    raise SystemExit(main())
