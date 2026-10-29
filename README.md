# MasterPrint: partial fingerprint security research

Complete offline CPU research implementation based on the supplied project
presentation. Includes **300 real SOCOFing original images**, the measured results,
one trained minutiae-feature Random Forest, raw score arrays, manifests, tests,
and an updated final presentation in the delivery folder.

## Start on Windows or Ubuntu

Use Python **3.12**. Extract this archive and open a terminal inside its
`masterprint` folder. No GPU, NBIS, API key or internet is required after installing
Python dependencies. Installation needs internet access unless wheels are cached.

```bash
python -m venv .venv
```

Ubuntu/macOS activation:

```bash
source .venv/bin/activate
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_all.py
python verify_results.py results/minutiae
python verify_results.py results/orb
```

The measured environment uses Python 3.12.14 and the pinned dependencies.
The two recorded runs took approximately 63 and 34 seconds on this host. Runtime
on your CPU can differ. To process just the primary implementation:

```bash
python run.py --data data/SOCOFing --n-identities 300 --matcher minutiae --target-fmr 0.01 --output results/minutiae
```

To score a local partial image using the trained research detector:

```bash
python detect.py data/SOCOFing/Real/1__M_Left_index_finger.BMP
```

The command above exercises inference using a full example image. For comparable
research probabilities, supply an image cropped with the same 70%-side protocol.
A full image or a new sensor is a distribution change. This probability is a
research proxy, not an authentication or liveness decision. Only load the trusted
`.joblib` files delivered with this project or trained by you.

## What the implementation contains

- Preprocessing, partial cropping, approximate minutiae extraction and local
  SIFT descriptors, alignment and geometric matching.
- ORB as a separate comparison matcher. SIFT keypoint matching is also available
  through `--matcher sift`, but no measured SIFT-only run is claimed here.
- Calibration at a specified **identity-level** FMR, including three-template OR
  enrollment, discrete EER and FNMR estimates.
- Breadth measurement across unrelated identities, with single-crop and
  multi-probe union coverage kept separate.
- A reference-gallery Random Forest and independent label gallery.
- A finite, detector-aware selection stress test over existing image crops.
- Deterministic seed, person-disjoint roles, provenance hashes, saved raw scores,
  trained primary model, plots and verification utilities.

`run_all.py` runs both matchers. Pass common arguments after it, for example
`python run_all.py --seed 43`, to explore another split. The supplied images support
at most 300 identities. Download the full original set before requesting more.

## Real data and provenance

The authors' SOCOFing dataset has 6,000 real originals from 600 people. This bundle
contains exactly the **300 original left-index images for people 1–300** used in
these experiments. Source files remain unchanged. Altered images are excluded.
Each result directory includes a manifest with subject ID, split and SHA-256.

For the complete 6,000 originals:

```bash
python download_data.py --output data/SOCOFing
```

This downloads the authors' approximately 878 MB Kaggle archive, extracts only
originals and validates the image count. It requires internet access and may be
subject to Kaggle availability. A manual alternative is to download the authors'
archive and place its `Real` directory under `data/SOCOFing`.
See **DATASET_NOTICE.md** for attribution, sources and noncommercial research terms.

## Experimental protocol

A seeded shuffle separates 300 people into:

| Role | People | Purpose |
|---|---:|---|
| Calibration | 50 | Freeze matcher threshold |
| Reference gallery | 50 | Produce detector input features |
| Label / discovery gallery | 50 | Define proxy labels and selection utility |
| Target gallery | 50 | Final coverage measurement |
| Detector training candidates | 50 | Fit forest, 250 crops |
| Validation candidates | 25 | Set detector cutoff and selection penalty, 125 crops |
| Test candidates | 25 | Evaluate, 125 crops |

There is one finger per person, so subject and finger identity coincide here.
No person crosses these roles. Each image yields five fixed crops: four corners
and the center, each 70% of original width and height. Gallery identities enroll
three corner crops. A probe accepts an identity if **any** enrollment crop passes.
Calibration compares center probes against those three-crop enrollments for all
50 × 50 identity pairs: 50 genuine and 2,450 impostor scores.

The matcher counts geometrically consistent correspondences after ratio filtering
and RANSAC. It rejects transforms outside scale 0.8–1.25 and requires at least
four inliers. Scores divide inliers by the smaller point count. The minutiae
baseline uses crossing-number ridge endings/bifurcations and local SIFT descriptors;
it is an approximate research matcher, not a validated commercial SDK.

The detector takes 12 features: mean, standard deviation, four quantiles, maximum,
top-five mean reference score, contrast, sharpness, foreground fraction and point
count. The **label gallery never supplies features**. Positive means matching at
least two label-gallery identities. The forest has 300 trees, balanced weights and
minimum leaf size 3. Validation negatives set its operating cutoff near 5% FPR.
Undefined AUC and rates return JSON `null`, never fabricated values.

Baseline selection ranks discovery coverage. Adaptive selection ranks
`discovery_coverage - lambda * detector_probability`. Validation selects lambda
from `[0, .05, .1, .2, .5, 1]` to minimize average detector probability while
retaining at least 80% of baseline validation discovery coverage. Then it freezes
lambda and selects ten test crops. Target-gallery scores never enter selection.
More than one crop from the same candidate person can enter the selected ten.

## Recorded results, seed 42

| Measure | Minutiae baseline | ORB comparison |
|---|---:|---:|
| Requested calibration FMR | 1% | 1% |
| Achieved calibration FMR | 0.898% | 0% |
| Calibration FNMR | 0% | 0% |
| Empirical crop-based EER | 0.265% | 0% |
| Detector test AUC | 0.808 | Undefined |
| Detector test TPR | 40.0% | Undefined |
| Detector test FPR | 5.263% | Undefined |

At this ORB threshold there are no positive training labels, so a valid detector
cannot be fitted. Its result report explicitly states this; no ORB trained model
is included. Zero observed calibration errors do not establish zero population
error. In particular, the genuine scores come from overlapping same-image crops.

For the primary matcher, the ten baseline and adaptive selected crops have mean
unseen-target coverage **4.0% and 3.8%**, respectively. Both sets have **28% union
coverage** across ten probes. Detector flag rate changes from **50% to 10%**.
The mean coverage retention is **95%**. These are distinct quantities and must
not be described as a 28% single-print success rate.

Across all 125 test crops, mean target coverage is **1.328%**, with a candidate-
person bootstrap 95% interval of **0.912%–1.824%**, conditional on the fixed target
gallery. The maximum single-crop coverage is **10%**. This is one exploratory
split; selected-set differences have no statistical significance claim.

## Outputs

Every `results/<matcher>/` contains:

- `report.json`: measured metrics, settings, environment and limitations.
- `config.json` and `manifest.json`: run settings and image hashes / role assignments.
- `scores.npz`: raw genuine/impostor and gallery scores, features, labels,
  detector probabilities, candidate groups and split roles.
- `candidates.csv`: row-aligned candidate identities, crops and measurements.
- `score_dist.png`, `coverage.png`: static experiment plots.
- `detector.joblib` and `detector_roc.png`: only if detector training is valid.

`verify_results.py` independently recomputes key reported metrics from saved
arrays, checks file hashes and checks candidate-person split separation.

## Limits that affect interpretation

1. SOCOFing has one original capture per finger. Genuine pairs are overlapping
   crops of the same image, which makes genuine performance optimistic.
2. This studies selected samples in an offline matcher. It does not generate
   prints, implement a GAN, perform physical presentation tests or establish
   bypass of a phone or other deployed product.
3. Broad matching against an independent gallery is a proxy label. This detector
   is not an independently validated MasterPrint detector or spoof detector.
4. Crops and comparisons share people. The simple AUC is crop-level, with no
   independent-sample significance claim. The supplied coverage interval resamples
   candidate people, not target identities.
5. Calibration has only 2,450 correlated impostor comparisons. Strong claims at
   rare FMRs require much larger independent-capture benchmarks.
6. One sensor distribution and one measured seed do not establish generalization.
   The code supports reruns, but additional runs must be reported separately.

## Source files

- `run.py`: experiment orchestration, calibration, training and selection.
- `masterprint/features.py`: templates, preprocessing and matcher.
- `masterprint/data.py`: strict original-image loader.
- `masterprint/metrics.py`: operating points, detector metrics and bootstrap.
- `detect.py`: local-image risk inference.
- `verify_results.py`: saved-result and provenance verification.
- `download_data.py`: full original-dataset acquisition.
- `tests/test_core.py`: numerical and feature edge-case tests.

Code: MIT. Dataset: its original noncommercial CC BY-NC-SA terms.
