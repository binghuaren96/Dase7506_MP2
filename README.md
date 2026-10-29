# DASE7506 MP2: Selecting Reliable LLM Reasoning

Select one of four supplied solutions with an answer-group linear
Bradley-Terry ranker. The frozen method uses 1024 group-mean embedding features
plus nine group/answer descriptors, with L2=0.01.

| Evaluation | Correct | Accuracy |
|---|---:|---:|
| Train out-of-fold, answer-disagreement questions | 1713/2114 | 81.03% |
| Complete validation | 742/798 | 92.98% |
| Answer-disagreement validation | 299/355 | 84.23% |
| Official test | **387/467** | **82.86937901498929%** |

Full validation includes 443 questions with only one valid answer group.
Test questions all have answer disagreement; the 355-question validation
subset is the closer comparison.

**Review version:** `review-v1`. The [eight-page report](report/report.pdf) and
[LaTeX source](report/report.tex) are ready for the project owner's review.
Repository publication does not itself submit a course result.

## Install and obtain data

Use Python **3.12** and an activated virtual environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on PowerShell or
`source .venv/bin/activate` on Linux/macOS. Then install either runtime only or
the training dependencies:

```bash
# Evaluation only: NumPy 2.5.3
python -m pip install -r code/requirements.txt
# Training and tests: also SciPy 1.18.1
python -m pip install -r requirements-training.txt
```

The unchanged course data is downloaded separately and SHA256-checked:

```bash
python code/fetch_data.py
python code/check_data.py
```

For an existing official course ZIP, use
`python code/fetch_data.py --archive /path/to/main.zip`. Matching installed
files are reused; conflicts are refused. Networking occurs only during data
installation. The selector works offline. No additional pretrained weights,
generated solutions, external dataset or API is required.

## Evaluate the supplied checkpoint

Set the thread limit before Python starts. PowerShell:

```powershell
$env:OMP_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = $env:MKL_NUM_THREADS = "4"
$env:BLIS_NUM_THREADS = $env:NUMEXPR_NUM_THREADS = $env:VECLIB_MAXIMUM_THREADS = "4"
```

Linux/macOS:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export BLIS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
```

Use the unchanged official prediction runner and independent scorer:

```bash
python code/predict.py --method code/method.py --checkpoint artifacts/final --split test --output runs/test-predictions.json
python code/score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

Expected: **387/467**, with `accuracy` in percent. Change `test` to `validation`
in both commands and output filenames for **742/798**. Neither command trains
or chooses parameters. For an instructor audit bank, prediction also accepts
`--split audit --data-dir /path/to/audit_inputs`; no IDs or sample counts are
hard-coded. Input and reference file roles follow the
[official course instructions](https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md).

## Reproduce training and the key comparisons

```bash
python code/train.py --config artifacts/final/config.json --run-dir runs/retrain
python code/train.py --config artifacts/final/config.json --run-dir runs/comparisons --reproduce-comparisons
python -m unittest discover -s code/tests
```

Output directories must be new. The first command performs one deterministic
fit from official train only and writes a new `model_bundle` plus one summary.
It filters 4755 questions to 2114 with at least two valid numeric answers.
It requires no saved labels, old runs, Git history or validation fitting.
The new bundle can be evaluated by replacing the checkpoint path above.

The comparison command performs a fixed **36 fits**: three representations
times three folds, followed by nine drop-one-feature models times three folds.
It rebuilds the exact fold algorithm on all 4755 train questions (seed 20261028),
then fits only informative questions outside the held-out fold. It writes one
`comparisons.json`; it does not access validation/test or replace the checkpoint.
Expected main OOF counts: candidate BT **1682**, group mean BT **1596**, and
group mean plus nine features **1713**, all out of 2114.

Final optimization is zero-initialized L-BFGS-B with fixed L2=0.01 and train-only
normalization. It has no stochastic seed, intercept or early stopping. The
comparison mode is a reproduction matrix, not a parameter search.
Numerical libraries can differ across platforms; the supplied frozen assets
remain the checkpoint associated with the official score.

## What is included

- `code/`: final inference/training, a fixed comparison mode, official runtime
  tools, verified data installation and focused tests.
- `artifacts/final/`: the single configuration and weights, **26,189 bytes** total.
- `experiments/`: a compact results table, the official score JSON and one
  metadata/resource record with seeds, optimizer settings and SHA256 hashes.
- `report/`: the eight-page report, source and AI/reuse disclosure.

Raw run folders, old models, predictions, internal plans, audit documents,
source snapshots and the course dataset are excluded. The historical results
table is a selected evidence record; old development frameworks are not needed
to train or evaluate the final method.

The original complete fresh-process test prediction plus scoring measurement
was **1.3019 s** on an Intel i7-12700H with four threads. This is local timing,
not reference-Xeon certification. The course limits are 30 s and 16 MiB.
See `experiments/metadata.json` for measurement scopes and reproduction results.

Validation was reused for development, and several frozen historical methods
were scored on the public test bank. These results are not described as a
single-use blinded test. The report separates controlled comparisons from
historical context and discusses this limitation.

## Attribution and report build

Keep [LICENSE](code/data/LICENSE.txt) and [NOTICE](code/data/NOTICE.txt) when
redistributing the package. GSM8K questions/references and the course-supplied
Qwen3-0.6B traces/representations retain their supplied attribution. The numeric
parser, scorer, runtime tools and data manifest are unchanged. See
[AI/reuse disclosure](report/ai-disclosure.md) and the
[official guide](https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md).

To rebuild the report with a standard TeX Live installation:

```bash
mkdir -p runs/report-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=runs/report-build report/report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=runs/report-build report/report.tex
```

On PowerShell, create the directory with
`New-Item -ItemType Directory -Force runs/report-build` instead of `mkdir -p`.
All references and disclosure within the PDF count toward its eight pages.
