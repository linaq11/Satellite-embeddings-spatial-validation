# Preregistration: do satellite embeddings keep their edge under spatial cross-validation?

Written on 2026-11-19, before any sample was drawn or any model was trained.

## Question

For crop-type mapping, do Google's AlphaEarth Foundations annual embeddings classify better than Sentinel-2 seasonal composites when only a few labels are available, and how much of each method's accuracy disappears when the validation is spatial instead of random?

## Data

* **Area.** California's southern San Joaquin Valley, the box from 120.6° W to 118.9° W and 35.8° N to 37.1° N (Fresno, Kings and Tulare counties).
* **Year.** 2023.
* **Labels.** USDA NASS Cropland Data Layer 2023 (`USDA/NASS/CDL`), only cells marked cultivated and with a CDL confidence of 90 or more. The classes are the 8 crop codes with the most such cells in the box; every other code is left out. The CDL is itself a classification, so these are reference labels, not field truth.
* **Sample.** 600 points per class, stratified random sampling at 30 m in Earth Engine with seed 42.
* **Features.** (A) the 64 AlphaEarth embedding bands for 2023 (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`); (B) Sentinel-2 surface reflectance, cloud-masked with the scene classification, median of each quarter of 2023 for 10 bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12), 40 features; (C) A and B together.

## Design

* **Validation.** (i) random 5-fold; (ii) spatial 5-fold, with points grouped in square blocks of 20 km and blocks assigned to folds at random (seed 0). Block sizes of 10 and 40 km are run as sensitivity checks, with the random forest only, feature sets A and B, and training sizes of 25 per class and all (scope fixed on 2026-11-19 after the sample was drawn and before any model was trained). Blocks are built in WGS 84 / UTM zone 11N. When a class has fewer training points than the requested size, all of them are used. Macro-F1 is computed over all 8 classes, with 0 for a class that the test fold lacks or that is never predicted.
* **Training size.** From the training folds, 10, 25, 50, 100 and 200 points per class, and all of them, drawn at random 10 times (seeds 0 to 9). The test fold is always used whole.
* **Models.** Random forest with 300 trees and default settings, and logistic regression on standardized features (C = 1, up to 2,000 iterations). Neither is tuned.
* **Metric.** Macro-F1 on the test fold.

## Hypotheses and decision rules

1. **H1, embeddings with few labels.** Under spatial validation, with 25 or fewer points per class, the random forest on A beats the random forest on B by at least 0.05 macro-F1. Decided on the mean of the paired differences (same fold, same draw) with a 95% bootstrap interval over the 50 pairs (5 folds by 10 draws) of each training size; H1 holds if the mean is at least 0.05 and the interval excludes zero at both sizes (10 and 25).
2. **H2, random validation inflates accuracy.** With all training points, random validation gives a higher macro-F1 than spatial validation for each feature set and model, by at least 0.02, with the 95% bootstrap interval of the difference excluding zero.
3. **H3, exploratory.** The advantage of A over B shrinks as the training size grows. Reported as a curve, without a test.

Every result is reported, including those that go against the hypotheses. Nothing in this design is changed after the first model is trained; any later change is listed as a deviation with its reason.
