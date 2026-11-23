# fNIRS subtyping of post-stroke aphasia: analysis code

This repository contains the methods code for the 79-patient fNIRS analysis described in the manuscript. It covers NIRS-KIT signal preprocessing, HbO feature extraction, similarity network fusion (SNF), spectral clustering and statistical comparisons. Study data and numerical results are not distributed here because of privacy and ethics restrictions. Data may be requested from the corresponding authors. Running this repository requires appropriately authorized local data; it is not a claim that the paper's numerical results can be reproduced without those data.

No license is supplied at this stage.

## Requirements

- MATLAB with NIRS-KIT for continuous-run preprocessing. Install NIRS-KIT separately; its code is not included.
- Python 3.11 or later. Install Python packages with `python -m pip install -r requirements.txt`.
- Input MAT files in MATLAB v7 format. Each must contain `nirsdata.oxyData`, `dxyData`, `totalData`, sample period `T`, and the pre-assessed `exception_channel` mask. Convert v7.3 files locally if necessary. Do not commit converted files.

## Input contract

All input and output paths must be outside this repository. Prepare the following locally:

| Input | Required fields / layout |
| --- | --- |
| Preprocessed recording directory | `<data>/<subject>/rest.mat`, `naming.mat`, `fluency.mat`; 110 channel columns per recording. |
| Cohort CSV | `subject,group`; unique subjects, 79 `PAT` and 17 `HC` rows. This file defines the analysis cohort. |
| ROI map CSV | `channel,roi`; channel indices begin at 1 and ROI indices are 1–22. ROI indices must follow the anatomical left/right order used in the manuscript. |
| Events CSV | `subject,run,condition,start,end`; optional `analysis_start,analysis_end` on every row of a run. Indices are zero-based, start-inclusive, end-exclusive. `run` is `naming` or `fluency`; conditions are `Naming`, `Phonemic`, `Semantic`. Include only reviewed, complete blocks. |
| Optional clinical CSV | `subject,age,post_onset_duration,education,WAB,BDAE,subtype`; optional numeric `sex` (0/1) for the secondary covariate models. |
| Optional behavioral CSV | `subject,naming,phonemic,semantic` for all 79 patients and 17 controls. |

Channel QC, event timing adjudication and selection of the 79-patient cohort must be completed locally before running the analysis. The code validates the provided inputs but does not infer these decisions from private records. Naming may contain four or five complete blocks; fluency requires five phonemic and four or five semantic blocks. For an incomplete terminal block, omit that block from the event table and set `analysis_end` to its adjudicated onset; the design regressors are filtered across the full continuous run before the analysis interval is selected. Use `analysis_start` for an adjudicated initial offset.

## Run order

1. Apply `matlab/preprocess_fnirs.m` separately to resting, naming and fluency recordings. Example MATLAB call: `preprocess_fnirs("D:/private/rest", "D:/private/processed/rest", "D:/tools/NIRS-KIT")`. Place the resulting MAT files in the per-subject layout above.
2. Extract the three clustering views and the 106 profiling features:

   ```text
   python -m fnirs_subtyping.cli extract --data D:/private/processed --cohort D:/private/cohort.csv --events D:/private/events.csv --roi-map D:/private/roi_map.csv --output D:/private/analysis/features
   ```

3. Run SNF, spectral clustering, 5,000 structured feature permutations, 1,000 patient subsamples, and the additional robustness refits:

   ```text
   python -m fnirs_subtyping.cli cluster --input D:/private/analysis/features --output D:/private/analysis/clustering --robustness
   ```

4. Run fixed-label patient–control and between-cluster profiling. Add `--clinical D:/private/clinical.csv` for clinical comparisons, subtype testing and patient-only covariate models; add `--behavior D:/private/behavior.csv` for four-group task scores, `--edges` for exploratory edge analyses, and `--clr` for mALFF sensitivity:

   ```text
   python -m fnirs_subtyping.cli analyze --input D:/private/analysis/features --labels D:/private/analysis/clustering/cluster_labels.csv --output D:/private/analysis/statistics --clinical D:/private/clinical.csv --behavior D:/private/behavior.csv --edges --clr
   ```

The full permutation and bootstrap analyses can take substantial time. For a quick software smoke check with authorized local inputs, use `--permutations 1 --subsamples 1`; those outputs are only a software check and must not be reported as study results.

## Methods map

| Manuscript method | Code |
| --- | --- |
| First-order detrending, TDDR, third-order 0.01–0.08 Hz filtering | `matlab/preprocess_fnirs.m` |
| 22 ROI time series, 180 s mALFF and rest FC; task GLM and rest–task FC | `fnirs_subtyping/features.py` |
| 781 features, seven family weights, locally scaled Gaussian affinity, SNF and deterministic spectral clustering | `fnirs_subtyping/clustering.py` |
| Eigengap null test, subsampling and robustness refits | `fnirs_subtyping/clustering.py` |
| Fixed-label profiling, FDR families, clinical comparisons and subtype test | `fnirs_subtyping/statistics.py` |

The three clustering views have 253, 66 and 462 features. Their row order is anchored to the cohort CSV. The profiling matrix has 106 features: 22 mALFF, 66 condition-specific activation and 18 anatomically grouped FC summaries. Python outputs are written only to the external `--output` directory.
