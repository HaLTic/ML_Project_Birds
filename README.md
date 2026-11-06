# COSC 434 — Group 12: Identifying Bird Species from Photographs

Mohamed Alromaithi and Zayed Alnuaimi.

## Simplified notebook: Version 2

Open `cosc434-group12-project-v2.ipynb` for the teaching version. It has ten
numbered stages, about 320 lines of Python, short explanations, and visibly
separate training cells for the four models. It is self-contained on Kaggle
and also runs locally from the project folder with the existing environment.
Attach CUB-200-2011 and either enable Internet or attach the official
`efficientnet_b0_rwightman-7f5810bc.pth` file. Paths can be set in Stage 1.

Version 2 keeps all 200 species, the same fixed splits, validation C selection,
training-only scaling, final metrics, confusion matrices, and image examples.
It replaces the automatic recovery, fingerprinting and extensive audit code
with straightforward sequential cells. It recomputes features and fits models
on each full run. Restart the kernel and run from the top after changing settings.
Its direct LinearSVC calls can be slower than Version 1's parallel wrapper.
Every classifier has the same explicit 2,000-iteration budget; unlike Version 1,
Logistic Regression does not retry at 10,000 iterations. Unconverged trials are
logged and excluded. These implementation differences are documented rather
than assuming the two versions always select identical models.

Outputs go into `results_v2/`; rerunning replaces files there. Copy that folder
to preserve a run. Version 1 and its existing results remain available for the
full dataset audit and resumable experiments described below.

Validation: the new notebook reproduced all original split IDs exactly.
A development run on 100 images (60 fit, 20 validation, 20 artificial evaluation)
across ten species passed all four models and 16 C trials, feature dimensions,
histogram normalisation, training-only scaling, prediction alignment, and
report/figure generation. The artificial evaluation images came from the
original training split, not the held-out project test set. Development files
are under `cache/v2_smoke_check.ipynb` and `cache/v2_smoke_results/`; their scores
are not project results. The full 200-species Version 2 model run and execution
of Version 2 on hosted Kaggle have not been verified.

## Original notebook: Version 1

`cosc434-group12-project.ipynb` implements the Part C experiment workflow in PyCharm or Kaggle:
data verification, fixed splits, a resource pilot, four model configurations,
validation tuning, final test evaluation, and error-analysis figures.
All 200 species are retained by default. Sections 1–7 prepare data;
Sections 8–15 run experiments.

## Upload and run on Kaggle

1. Import the updated `cosc434-group12-project.ipynb` into your Kaggle notebook.
2. Use **Add Input** to attach CUB-200-2011. The existing input named `NA_Birds`
   can be kept if it contains `CUB_200_2011.tgz`, as shown in your earlier
   extraction cell. The optional NABirds collection itself is a different dataset.
3. Enable Internet for the first official EfficientNet weight download, or attach
   `efficientnet_b0_rwightman-7f5810bc.pth` and set `WEIGHTS_FILE` in Section 8.
   A GPU is recommended for CNN extraction; CPU also works.
4. Choose **Run All**. The first cell should print `Runtime: kaggle`, with
   `/kaggle/input` as the search folder and `/kaggle/working` as the output folder.
5. Review the split counts and final experiment results. Use **Save Version** to
   retain the run and its output files.

The notebook is self-contained: no local Python modules, README, or requirements
files need to be uploaded. Use Kaggle's installed packages; the Windows lock file
is for the local environment. Data preparation makes no network requests;
the first CNN run downloads official weights unless supplied. Optional package-version reporting does not fail if
desktop packages such as JupyterLab are absent.

Archives are discovered recursively under `/kaggle/input`, regardless of the
dataset slug. Supported names are `CUB_200_2011.tgz`, `BirdImages_Dataset.tgz`,
and their `.tar.gz` equivalents. If multiple matching sources are attached,
set `DATASET_SOURCE` in the first code cell to the intended archive or extracted
folder. If a complete extracted dataset is attached, the notebook reads its
images and metadata directly without writing to the input mount.

Generated splits and reports are written to `/kaggle/working/splits` and
`/kaggle/working/results/data_checks`. When an archive is used, extracted images
and the preview are also saved under `/kaggle/working/data`.

Data-preparation portability was checked locally with simulated Kaggle input/working directories
for both a nested archive and a pre-extracted dataset. Both complete runs
reproduced all four local manifest files byte-for-byte. Missing/ambiguous input
handling and optional package reporting were also checked. These checks do not
substitute for executing the notebook in your hosted Kaggle session.

## Local environment and execution

Validated environment: Windows x64, Python 3.13.14. The notebook prints the
actual Python executable and package versions in its first code cell.
`requirements.txt` pins direct dependencies; `requirements-lock.txt` records
the complete installed Windows environment, including notebook tooling.

For a new checkout, create a Python 3.13 environment from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

The existing project environment is already installed. In PyCharm:

1. Select the project interpreter `.venv/Scripts/python.exe`.
2. Open `cosc434-group12-project.ipynb` and use that environment for its kernel.
3. Start the notebook with the project directory as its working directory.
4. Run all cells from top to bottom. To verify a clean execution, restart the
   kernel first, then run all cells.

On another operating system, use `requirements.txt` rather than the
Windows environment lock and verify the regenerated split manifests.
The data stage runs on CPU. The CNN automatically uses CUDA when available.
The local dependency lock records the validated CPU installation; use Kaggle's
provided CUDA-enabled PyTorch rather than replacing it with the Windows lock.

## Obtain the dataset

Download **Images and annotations** from the
[official CUB-200-2011 page](https://www.vision.caltech.edu/datasets/cub_200_2011/).
Locally, place the archive in the project directory with either name:

- `BirdImages_Dataset.tgz` (the existing local file).
- `CUB_200_2011.tgz` (the official filename).

For archive inputs, the notebook extracts the `CUB_200_2011` directory into
the writable output root's `data/` folder, using
Python's data extraction filter. It records the archive SHA-256 and writes
`data/extraction_complete.json` only after extraction succeeds. Re-running
with the same archive skips extraction but repeats the metadata/image checks.
An interrupted extraction can be rerun. A different archive is rejected when
a previous completion marker exists, so dataset versions cannot be mixed
silently. The recorded hash identifies the local input; it is not a claim
that the file has been authenticated against a publisher-provided checksum.

For an already-extracted input, `archive_sha256` is recorded as `null` because
the original archive is unavailable. Metadata hashes and individual image hashes
are still recorded. The `source` field in the preparation report identifies
which input representation was used.

The publisher restricts use to non-commercial research and educational
purposes and does not own the image copyrights. Raw archives, extracted images,
and the local preview are excluded by `.gitignore`. Distribute code, split
manifests, and download instructions subject to the publisher's terms. The
notebook's displayed training preview also contains dataset photographs.

## Checks and fixed splits

The notebook joins `images.txt`, `image_class_labels.txt`,
`train_test_split.txt`, and `classes.txt` by their IDs. It verifies:

- 11,788 unique image IDs and paths, complete labels, and 200 species.
- The supplied 5,994 training / 5,794 test image separation.
- Agreement between each class directory and the corresponding metadata label.
- Readability and RGB conversion of every image; no images are silently dropped.
- Disjoint split IDs and no byte-identical image files across splits.
- All species present in each split and successful readback of saved CSVs.

Only the official training portion is split with
`train_test_split(test_size=0.20, stratify=class_id, random_state=42)`.
Image records are sorted by ID before splitting. The official test set is
preserved exactly.

| Split | Images | Species | Purpose |
| --- | ---: | ---: | --- |
| Training | 4,795 | 200 | Fit feature scalers and classifiers |
| Validation | 1,199 | 200 | Select settings using macro-F1 |
| Test | 5,794 | 200 | Final evaluation after settings are fixed |

The CSV manifests retain the canonical project-relative paths and original
class IDs 1–200. Always open images with `image_file(row.relative_path)`, which
resolves the same manifest against either extracted local storage or the actual
read-only Kaggle dataset folder. This keeps the manifests identical between
environments without embedding platform-specific paths.
Every model must use these same manifests. Existing CSVs are compared against
regenerated assignments; different assignments cause an error instead of
silently overwriting the saved split.

Image decoding and file-hash checks include test files only to check data
integrity. The displayed examples are exclusively training images. Exact file
hashes do not identify every visually similar or re-encoded duplicate.

The verified archive contains 11,780 RGB images and eight grayscale images;
all decode and convert to RGB successfully. One exact duplicate pair (image IDs
3310 and 3340, Pigeon Guillemot) is entirely within the training split. Both
records are retained to preserve the specified dataset and fixed assignments;
this slightly repeats one training example. No exact duplicate file crosses
training, validation, or test. These findings are recorded in the audit outputs.

## Files produced

| File or folder | Contents |
| --- | --- |
| `data/CUB_200_2011/` | Writable extraction of an archive; unused for an already-extracted input |
| `data/training_examples.png` | Nine labelled training examples |
| `cache/` | Feature arrays and downloaded pretrained weights |
| `splits/image_manifest.csv` | All images, labels, paths, official splits, and assigned splits |
| `splits/train.csv`, `validation.csv`, `test.csv` | Fixed subsets |
| `splits/split_config.json` | Seed, split settings, versions, counts, input and manifest hashes |
| `results/data_checks/image_audit.csv` | Dimensions, original modes, and image file hashes |
| `results/data_checks/image_errors.csv` | Unreadable/missing image report; empty when checks pass |
| `results/data_checks/exact_duplicate_images.csv` | Exact duplicate file groups, if any |
| `results/data_checks/class_counts.csv` | Per-species counts in each split |
| `results/data_checks/preparation_summary.json` | Checks, environment, CPU context, and measured preparation times |

The preparation report is refreshed on each run. Extraction time will be
shorter when the dataset is already extracted. These are preparation timings,
not model training or feature-extraction benchmarks.

## Experiment execution and outputs

Run Sections 8–15 after data preparation. The notebook measures extraction time
and a process-memory snapshot on training samples, then runs these configurations:

1. Normalised RGB colour histogram + Logistic Regression.
2. HSV colour histogram + training-only StandardScaler + Linear SVM.
3. The same HSV histogram + uniform LBP texture + training-only StandardScaler
   + Linear SVM.
4. Frozen ImageNet-pretrained EfficientNet-B0 features + training-only
   StandardScaler + Logistic Regression.

Section 8 defaults to `EXPERIMENT_SPECIES = 200`. Full 200-species SVM tuning
can take substantial CPU time; a GPU accelerates CNN extraction, not the SVMs.
The proposal's optional resource fallback is available with
`EXPERIMENT_SPECIES = 50`: the first 25 warblers, 17 sparrows and 8 gulls in class-ID
order. This taxonomy-only choice retains confusable species. It never depends
on test performance. Original image assignments are retained and filtered;
each scope saves its own `experiment_manifest.csv`, settings, caches and results.
Report the chosen scope explicitly and justify a 50-species project run using
resource measurements. Its scores must not be presented as 200-species results.

The controlled investigation compares configurations 2 and 3 using identical
splits and the same validation tuning procedure. Fit preprocessing statistics
on training data only. Select classifier regularisation using validation
macro-F1, then evaluate fixed models once on the test set. Report accuracy,
macro precision/recall/F1, runtimes, and qualitative error analysis.

The publisher warns that CUB images may overlap ImageNet pretraining images.
The notebook discusses that limitation and its use of additional training data.
NABirds remains an optional later extension.

RGB uses 16 marginal bins per channel (48 features); HSV uses an 8×8×8 joint
histogram (512). Uniform LBP uses 8 neighbours, radius 1, and 10 bins. Each
histogram is normalised to sum to one. Full images are resized to 128×128 with
bilinear interpolation. The frozen CNN uses the official weights' aspect-preserving
resize, centre crop, and ImageNet normalisation. This is a documented refinement
of the square CNN resize described in the proposal. CNN features have 1,280 dimensions.

Each classifier tries C = 0.01, 0.1, 1, 10; highest validation macro-F1 wins,
with smaller C breaking ties. Scalers are fitted inside training-only pipelines.
Both SVM configurations use primal LinearSVC with a one-vs-rest wrapper. Up to
eight shared-memory threads fit independent species classifiers within the
notebook kernel; no worker Python processes are launched. This preserves the
same linear SVM objective while reducing fitting cost. Settings record the solver
and thread count.
SVM trials have a fixed 2,000-iteration budget; Logistic Regression can retry at
10,000 iterations. Unconverged trials are logged and excluded from selection.
If no setting converges for a configuration, evaluation stops. Both SVM models
use the same grid, budget, and convergence rule. Models are not refitted on validation
data. Test results are generated only after all four choices are saved.

`results/experiments/<run-id>/` contains:

- `settings.json`, `hardware.json`, `pretrained_resource.json`: settings,
  software/hardware, pretrained provenance and checksum.
- `resource_pilot.csv`, `feature_runtimes.csv`: pilot measurements, extraction
  times and current cache-load times. Process RSS is not peak memory.
- `*_validation_trials.csv`, `validation_selection.csv`: all trials and winners.
- `models/`, `locked_selection.json`: fitted classifiers, scalers and hashes.
- `test_predictions.csv`, `test_evaluation_receipt.json`: reusable final predictions.
- `test_metrics.csv`, `model_comparison.png`: the four-model comparison.
- `*_confusion_full.csv`, `*_per_class.csv`, `*_warbler_confusion.png`:
  complete confusion/per-class reports and readable plots.
- `top_confusion_pairs.csv`, `example_review.csv`, `results_summary.txt`:
  error-analysis evidence, a preserved worksheet for visual observations, and
  the automatically computed score summary.

Prediction-example grids contain dataset photographs and are stored under
`data/*_prediction_examples.png`. Review them and add pose/background/plumage
observations to `example_review.csv` before preparing the presentation.

Run All from a fresh kernel to reproduce execution. Caches and models are keyed
by settings, package versions, split hashes and the image audit. Preserve
`cache/features/<run-id>/` and the matching results folder to resume. Saved test
predictions are reused only if their model hashes match. For a completely fresh
run, use a new working copy with the same archive, environment, and manifests.
Do not change methods in response to final test scores. GPU and CPU numerical
results may differ slightly; compare using the recorded hardware and versions.

## Implementation verification (6 November 2026)

The complete experiment code passed a local CPU run on the documented
50-species subset: 1,199 training, 300 validation and 1,464 test images.
A fresh-kernel rerun restored the models and final predictions successfully.
Checks verified all 16 tuning trials, training-only scaler statistics,
prediction IDs, complete confusion matrices, figures and prediction checksums.
The tested experiment code matches the upload notebook except for the species
count. Verification artifacts are in `results/experiments/664159fff7bf8f25/`.

Four SVM trials (C=1 and C=10 for each SVM) did not converge within the fixed
budget and were excluded as specified. Every selected model converged.
The 50-species test macro-F1 scores were 0.0326 (RGB), 0.0438 (HSV),
0.0535 (HSV+LBP), and 0.7612 (EfficientNet). These are verification-subset
results, not results for all 200 species.

The upload notebook still defaults to 200 species. Its full model-training run
and the new experiment sections in hosted Kaggle have not yet been verified.
Run All on Kaggle, retain the resulting outputs, and review the actual error
examples before finalising the project discussion and presentation.

## References

- Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. (2011).
  *The Caltech-UCSD Birds-200-2011 Dataset*, CNS-TR-2011-001.
  [Dataset and technical report](https://www.vision.caltech.edu/datasets/cub_200_2011/).
- [scikit-learn: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
- [scikit-learn: preventing preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).
- [Kaggle's Python environment](https://github.com/Kaggle/docker-python).
- Group 12 proposal (v18) and the COSC 434 Fall 2026 project brief/deliverables
  supplied with this project.
