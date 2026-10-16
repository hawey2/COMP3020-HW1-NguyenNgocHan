# 1. Setup and rules

**Goal:** get a working environment, confirm you have the right data file, and know which tools you may use.

## What to do

1. Use **Python 3.10 or newer**. The starter code uses modern type annotations.
2. Create a virtual environment and install the packages. Run this from the assignment directory:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Read [DATASET.md](../DATASET.md) for the column dictionary, local file statistics, and interpretation caveats.
4. Load the **local** file. Do not download a replacement:

   ```python
   import pandas as pd

   df = pd.read_csv("dataset.csv", encoding="utf-8-sig")
   ```

   The file is comma-separated and has a UTF-8 byte-order mark. `main.load_dataset` already does this and assigns stable row IDs.
5. Run `python main.py --inspect` and check the output against the table below.

## Check yourself: expected data facts

| Item | Expected value |
| --- | ---: |
| Student records | 4,424 |
| Predictors, excluding `Target` | 34 |
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled | 794 |
| Empty cells | 0 |

Also do the following, and record what you find:

- [x] Keep a stable row ID taken from the original row position. Use it for splitting, tie-breaking, and explanations. **Never** use the row ID or `Target` as a predictor.
- [x] Check duplicates, column types, value ranges, and category frequencies. **Document** anomalies; do not silently delete rows.
- [x] Keep the spelling `Nacionality` in code, or explicitly document any rename.

> **Note: category codes are identifiers.** The local `Course` values are 1–17. The [UCI documentation](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) uses other identifiers and includes predictors that are not in this CSV. Do not copy its code-to-name mappings. Treat local category meanings as unverified unless you find a matching codebook. Course-rate tables may use the local codes. Regrouping education codes is optional and needs a verified mapping; keeping the codes nominal is enough.

## Allowed and forbidden tools

| You may use | For |
| --- | --- |
| NumPy, pandas | Calculations and data handling |
| Matplotlib | Plots |
| scikit-learn | Splitting, preprocessing, and evaluation **only** |
| Standard-library `unittest` | Algorithm checks (no notebook is required) |

You must **not** use:

- scikit-learn's tree, neighbor, nearest-neighbor search, or distance implementations;
- ready-made classifiers from any other library.

Write these yourself: entropy, information gain, tree construction and traversal, distances, neighbor selection, and risk scoring.

## Commands

Run these from the assignment directory:

```bash
python main.py --inspect                                          # works immediately
python -m unittest discover -s tests -v                           # full test suite
python -m unittest discover -s tests -p 'test_decision_tree.py' -v
python -m unittest discover -s tests -p 'test_knn.py' -v
# After completing all TODOs:
python main.py --data dataset.csv --output results --seed 42      # full experiment
```

To run only the two provided scaffold checks before you implement anything:

```bash
python -m unittest discover -s tests -p 'test_main.py' -k TestProvidedScaffold -v
```

### What to expect at the start

- `--inspect` works immediately.
- The full experiment command reports the unfinished TODO and exits with status 2 until you implement it.
- Only the provided CSV/schema checks pass at first. The implementation tests are **meant** to fail on the untouched skeleton.

### Test rules

- **Do not** skip the supplied tests, mark them as expected failures, or change their expected answers to get a passing suite.
- **Do** add checks for any new behavior you introduce.
- Passing the supplied tests shows correctness on small examples. It does **not** mean you have finished the experiment or report requirements.

Next: [Milestones](02-milestones.md).
