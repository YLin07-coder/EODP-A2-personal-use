# Assignment 2 Group Report: Predicting recorded Superhost status from listing and host attributes
## COMP20008 Elements of Processing, Semester 2, 2026

### Group Information
Group Name: W01G03
Tutor: Kenny Lam

Group Members:

| Name       | Student ID | Email                             |
| ---------- | ---------- | --------------------------------- |
| Yilin Wang | 1810022    | ywang12825@student.unimelb.edu.au |
| Yili Liu   | 883012     | yilil@student.unimelb.edu.au      |
| Enoch Ko   | 147938     | enoch.ko@student.unimelb.edu.au   |
### Research Question
Can ‘host_is_superhost’ be predicted from listing and host attributes, and which features matter most?

> **FOR ENOCH ONLY: formatting.** Calibri 11 pt body, single-column A4, normal margins, line spacing exactly 1. Tables use 9 pt; captions use 9 pt, above tables and below figures. Australian English and APA references. Markdown has no fixed pagination: the allocation below is a writing target, not a verified nine-page layout. Avoid a separate full-page title sheet.

## 1. Introduction

The research question chosen was: "Can `host_is_superhost` be predicted from listing and host attributes, and which features matter most?"

The prediction methods considered were K-Nearest Neighbours (KNN) and Decision Tree (DT), tested against Baseline (0R) using an unseen host-disjoint test set (i.e. same host does not exist in both training and test data set).

- Writing reminders: interpret the top-three tree/MI features; compare removal of review fields; distinguish prediction from causes of Superhost status.

For this research question, the Inside Airbnb Melbourne dataset labelled 16 June 2026 (Inside Airbnb, 2026). This dataset contains 25,728 listings, 90 raw columns, 14,113 distinct hosts. `host_is_superhost` contains 7,885 positive listings (30.6%) and 17,843 negative listings (69.4%).

The models were tested against unseen hosts within this dataset, and may not represent future superhost eligibility.

## 2. Methodology

### 2.1. Data preprocessing

The raw dataset contains 25,728 listings and 90 columns. We constructed a modelling table with 26 predictors from 19 source fields.

The preprocessing was organised around three decisions that could materially affect the information available to the models:

- review recency
- absent review information
- missing price or room-size information

Other transformations were supporting conversions that simply represented existing fields numerically.

**Review recency.** The `last_review` field was parsed as a date. We used the latest review date observed in the file, 28 June 2026, as the reference date and calculated `days_since_last_review` as the number of days between each listing's last review and that date. This produced a recency value for 21,251 listings. The remaining 4,477 listings had no recorded review and therefore no review date. Using the latest date in the file rather than the notebook execution date makes the transformation reproducible and avoids negative values for listings reviewed on the latest observed date.

**Absent review information.** We retained the 4,477 never-reviewed listings (17.4%) rather than deleting them. A binary `never_reviewed` indicator identifies these rows. Review-related numerical fields—`days_since_last_review`, `reviews_per_month`, and the three review scores—remain missing at this stage and are imputed later using medians calculated from the training data. The indicator is retained so that the model can distinguish an imputed value from an observed review value. This avoids treating a missing review history as equivalent to zero review activity.

**Missing price and room size.** We retained 9,960 listings (38.7%) with at least one missing value among price, bedrooms and beds. The cleaned price removes currency symbols and thousands separators and is converted to a numerical value. `price_missing`, `bedrooms_missing` and `beds_missing` record missingness explicitly. During training, bedrooms and beds are imputed using the median within `room_type`, with the overall training median as a fallback; price uses the overall training median. This keeps incomplete listings in the modelling population while avoiding imputation from the test set.

The remaining transformations convert existing fields into numerical predictors:

- bathroom text becomes a numeric bathroom count
- the amenities JSON becomes an amenity count
- host verification is encoded as 0/1
- latitude and longitude become distances in kilometres from a fixed Melbourne CBD reference point (City of Melbourne, 2026), using a spherical Earth approximation (Moritz, 2000)
- `room_type` becomes four binary indicators
- `host_listings_count` is retained, with an `is_multi_listing_host` indicator, so that host operating scale can be compared with the earlier trial run

> **FOR ENOCH ONLY: CBD-distance calculation. Convert the points below into Section 2.1 prose here.**

- EdStem #175 permits outside constants but asks us to explain their purpose and effect.
- Identify the fixed reference coordinates (-37.813811, 144.962915), representing the City of Melbourne's Melbourne (CBD) CLUE Geo Point, rounded to six decimal places.
- Explain that haversine angular distance is multiplied by the mean Earth radius of 6371.0088 km to produce distance in kilometres.
- Retain the existing City of Melbourne and Moritz citations.
- These details are already in notebook Section 3; no new calculation is needed.

We considered three additional representations but did not use them:

- **Property-type consolidation.** The dataset contains 82 property types, and `room_type` already provides a simpler representation.
- **Minimum-stay bands.** Only 426 listings (1.7%) have a minimum stay of at least 30 nights, so bands would discard the numeric value.
- **CBD-distance bands.** The median distance is 5.11 km. Distance was retained as a continuous variable rather than cut at arbitrary boundaries.

> **FOR ENOCH ONLY: preprocessing choices to explain. Incorporate these outline points into Section 2.1.**

- Make the six principal candidates explicit: selected — review recency, absent-review handling, missing price/room-size handling; rejected — property-type consolidation, minimum-stay bands, CBD-distance bands. Keep supporting conversions separate.
- Review recency:
  - Alternative: omit recency and retain review frequency/counts alone.
  - Reason to retain it: 21,251 listings have usable dates; dropping recency lowers fixed-setting validation macro-F1 by 0.0059 KNN and 0.0133 DT (Table 3).
  - Measurable change: dates become numeric elapsed days for 21,251 listings; the other 4,477 remain explicitly identifiable as never reviewed.
- Absent review information:
  - Alternative: delete the 4,477 never-reviewed listings (17.4%); this would restrict the RQ to reviewed listings.
  - Alternative tested: zero-fill only their review frequency while retaining the flag; changes are +0.0002 KNN and -0.0001 DT (Table 3), giving little evidence of benefit over median filling.
  - Reason for retaining the flag: missing history differs from an observed median; flag removal reduces KNN validation macro-F1 by 0.0165, while DT is unchanged (Table 3).
- Missing price/room-size information:
  - Alternative: delete the 9,960 incomplete listings (38.7%); this would substantially narrow the population being predicted.
  - Explain why room-type medians are used for bedrooms/beds rather than a single overall median: room type supplies context for size, with an overall fallback. This is a modelling rationale, not a measured comparison of imputation methods.
  - Before/after missing counts are in Table 1; acknowledge that filling values preserves rows but cannot recover their true values.
  - Column removal in Table 4 is a separate joint-removal check, not a test establishing that median imputation is optimal.
- Rejected candidates: connect 82 property types, 426 long-stay listings (1.7%) and median CBD distance 5.11 km to representation choices; do not imply that these numbers alone prove the rejected transformations would perform worse.

The same processed table is later trained under two nested conditions, described in Section 2.3.1. Those conditions compare reliance on review information. They are not alternative preprocessing pipelines.

> **FOR YILI ONLY:** Retain a short scope limitation in section 2.1: the 26 predictors from 19 source fields are a designed candidate set, not an exhaustive search of all 90 columns. The old candidate-list reminder has already been removed. Background: [field selection evidence](reference/rq1-column-scope-notes.md).

### 2.2. Correlation analysis
Set C (Host activity): `host_listings_count`, `days_since_last_review`, `reviews_per_month`, `review_scores_rating` were correlated with the binary Superhost target.

Host count describes the hosts' operating scale. Review recency and frequency describe review activity. Rating describes guest feedback. Compare each feature with recorded Superhost status allows us to identify predictive associations.

Due to the inclusion of reviews, we use the common complete-case population of 21,251 reviewed listings, excluding 4,477 never-reviewed listings.

All ten pairs were correlated using all four required methods:
  - Pearson: linear association with the binary target.
  - Spearman: rank association, which is useful for skewed count/recency variables.
  - Mutual Information (MI) and Normalised Mutual Information (NMI): ten rank intervals for numeric predictors while keeping ties together, meaning the occupied bins can be fewer and uneven in size (host count 8, rating 9), with arithmetic normalisation used for NMI. The target remains binary. MI is in nats (scikit-learn uses natural log), is unsigned, and is not on the correlation scale, so its magnitude cannot be compared directly with Pearson or Spearman.

> **FOR ENOCH ONLY: correlation implementation choices to explain.**

- Explain the ten-bin resolution against the 21,251 reviewed rows: about 2,125 observations per intended rank interval before ties; a balance between resolution and contingency-table sparsity.
  - This is a rationale for the chosen resolution, not evidence that ten bins are optimal; alternative bin counts were not compared.
  - Keep identical values together so an arbitrary ordering cannot split equal measurements into different categories; this produces 8 occupied host-count bins and 9 rating bins.
  - Average observations per interval do not guarantee well-populated joint cells, particularly with ties and dependent predictors.
- Explain binary target coding: 0 = recorded non-Superhost, 1 = recorded Superhost. Pearson with this binary target is a point-biserial association; Spearman compares predictor ranks with the two target groups, retaining ties. Neither reconstructs eligibility rules.
- State why MI/NMI use binned numeric predictors while the target stays binary; distinguish these contingency-table scores from the univariate MI estimator used for feature selection in Section 2.4.

The correlation analysis was run twice. Once as a descriptive all reviewed listings table, and another time as separate training-only correlations for modelling decisions to ensure the test rows did not inform modelling decisions.

Since many hosts own several listings, and superhost status belongs to the host, the listings are not independent, and p-value would overstate significance. Thus significance test were not run and p-values are not reported.

### 2.3. Supervised learning models and evaluation

#### 2.3.1. Nested feature conditions

The main model uses the Full condition: all 26 predictors from the processed feature table.

> **FOR ENOCH ONLY: incorporate into Section 2.3.1 prose.** EdStem #224 allows modelling features to differ from the correlation set, but requires their relevance to the research question to be justified. Briefly explain that the 26-predictor candidate set covers host scale/verification, listing structure/operations/location, and review information. Keep the qualification that this is a designed candidate set, not an exhaustive search of all 90 raw columns. Background: [field selection evidence](reference/rq1-column-scope-notes.md).

As a secondary check, we also train on one smaller feature set. The specification does not require this comparison. We include it to see how well the non-review features predict on their own. Both conditions start from the same processed table and use the same imputation methods.

- **No review** removes the three guest scores (`review_scores_rating`, `review_scores_cleanliness`, `review_scores_value`) and review activity (`number_of_reviews`, `reviews_per_month`, `days_since_last_review`, `never_reviewed`) from Full. This leaves 19 predictors covering host attributes, listing structure, amenities, location, stay/availability and missingness flags.

#### 2.3.2. Shared modelling procedure

The same procedure below is used for both conditions. The folds and the hyperparameter grid are shared. The chosen setting is selected separately for each condition.

We hold out the first of five shuffled stratified host-grouped folds (with random seed 20008) as test data. This results in a training data set of 20,582 listings from 11,292 hosts, of which there are 6,308 listings with Superhost status (30.65%). The test data set has 5,146 listings from 2,821 hosts, of which there are 1,577 listings with Superhost status (30.65%). There is no host overlap between training and test data - this protects against learning the same host on both training and test data.

We then use five stratified host-grouped training folds for tuning, with identical folds across conditions and models.

> **FOR ENOCH ONLY: split and validation justification to add.**

- Connect stratification to the 30.65% Superhost share in both splits; a majority-only predictor can appear accurate while failing to identify Superhosts.
- Explain grouping over a listing-level random split: 25,728 listings belong to 14,113 hosts, and the RQ concerns performance on unseen hosts; keeping a host together avoids sharing that host across training and evaluation.
- Justify five training folds using actual validation sizes: 4,116–4,117 listings, 2,258–2,259 hosts and 30.64–30.66% Superhost per fold (source: `notebook_outputs/tables/cv_fold_balance.csv`).
  - Each fold retains substantial minority-class representation while allowing repeated validation across the training data.
  - Discuss the trade-off against one validation split and more folds: repeated checks versus tuning cost and smaller validation groups; do not claim alternative fold counts were experimentally compared.
- Explain why this release supports an unseen-host split; a later-release temporal test would answer future generalisation and remains an improvement opportunity.

> **FOR ENOCH ONLY: modelling-specific preprocessing. Convert the points below into prose here.**

- EdStem #238 asks for modelling-specific preprocessing to be briefly described in Supervised Learning Methodology.
- Fit imputation separately on the training portion of each tuning fold; apply it to that fold's validation rows.
- For final models, fit imputation on the full training split and apply it to the test split.
- KNN scaling follows the same training-only rule.
- Mention the four binary room-type indicators; cross-reference Section 2.1 for detailed missing-value strategies.
- These points document the current notebook procedure; no pipeline change is needed.

For KNN, we used k = 1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, with scaled and unscaled inputs. Scaling for KNN uses `StandardScaler`, which standardises each column by subtracting the training mean and dividing by its training standard deviation. Beyond that, KNN function `KNeighborsClassifier` was left on defaults uniform weights, Euclidean distance with default Minkowski p = 2.

For Decision Tree, we used depths 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, and unbounded, minimum leaf sizes 1, 5, 10, 20, 50, and random seed 20008. Beyond that, DT function `DecisionTreeClassifier` was left on defaults using Gini criterion.

For each model (KNN/DT) and feature condition (full/no review), we selected the hyperparameter combination with the highest mean validation macro-F1 across the five training folds.

- Metrics and reference:
  - Macro-F1 weights the two classes equally; connect this to the 69.4% non-Superhost majority and 0R's zero Superhost recall (Table 12).
  - Per-class precision and recall answer how often predicted Superhosts are correct and how many actual Superhosts are found; the never-reviewed subgroup shows why high accuracy alone is insufficient (Table 11).
  - ROC-AUC assesses ranking; predictions use each classifier's default decision rule.
  - 0R predicts the training majority class; same test rows and metrics.
- Uncertainty:
  - 1,000 bootstrap resamples of whole test hosts; shared draws for paired model/condition differences; percentile 95% intervals.
  - Preserve within-host clustering: listings from the same host share a target and need not be independent; resample the 2,821 test hosts rather than individual listings.
  - Justify bootstrap for uncertainty in the existing fitted models' test scores: no model retraining or normality assumption; shared draws estimate paired differences on the same hosts.
  - Contrast with repeated/nested grouped CV, which would address training/split/tuning variation. These intervals condition on fitted models and do not measure that broader uncertainty.
- Sensitivities:
  - Reviewed/never-reviewed and collection-date subgroups.
  - Training-fold comparisons with chosen hyperparameters held fixed; exploratory, not independent performance confirmation.

> **FOR ENOCH ONLY: evaluation scope and column-removal checks. Convert the points below into prose here.**

- The same release was examined during the earlier exploratory trial; the holdout is not an untouched prospective sample.
- Predictions concern recorded status, not a reconstruction of Airbnb eligibility.
- Listing review counts do not measure completed reservations or recover all host-level eligibility quantities and assessment windows.
- The exploratory column-removal checks drop price, beds and their two missingness flags together; they cannot isolate those components.

### 2.4. Feature selection

- Embedded method: impurity importances from the full-feature fitted tree.
- Filter method: univariate MI against target on the imputed training matrix.
  - Counts/indicators explicitly discrete; fractional bathroom levels category-coded.
  - Rating, price, distance and frequency continuous; fixed estimator seed.
- Compare top three and one disagreement outside the top three.
- Hard case: held-out listing misclassified by both full models and in a tail of the top tree feature.
  - Post-evaluation explanatory case, not another tuning decision.

> **FOR ENOCH ONLY: hard-case selection. Convert the points below into Section 2.4 prose here.**

- EdStem #237 asks us to briefly explain how the hard case was selected.
- After evaluation, consider test listings misclassified by both Full models.
- Among these, identify listings whose observed review frequency, the tree's top-ranked feature, is at or below the 5th percentile or at or above the 95th percentile of all observed values.
- Select the lowest listing ID among these tail cases.
- The percentile is the share of observed values at or below the listing's value.
- This selects listing 16261169, with 4.32 reviews/month at the 95.46th percentile.
- Keep the distinction between post-evaluation explanation and model tuning.
- Source: notebook Section 10 and `notebook_outputs/tables/hard_case_candidates.csv`.

> **FOR ENOCH ONLY: feature-selection interpretation and cautions. Convert the points below into Section 2.4 prose here.**

- Filter MI scores each predictor separately.
- Tree importance totals impurity reductions from splits conditional on earlier partitions; a feature not used in a split receives zero importance.
- Impurity importance may favour predictors with many possible split points.
- MI estimation depends on discrete/continuous treatment and finite sample size.
- These are properties of the methods, not biases or ranking changes measured in our results.

> **FOR ENOCH ONLY:** Omit sections 2.5, 3.5 and 4.5 for PCA/clustering. They apply only to groups of four. Do not add a third classifier merely to increase the number of techniques.

## 3. Results Exploration and Analysis

### 3.1. Data preprocessing

All 25,728 rows from the input file `listings.csv` were retained. The retained dataset contains 14,113 distinct hosts.

Recency `days_since_last_review` was created for 21,251 listings, with the other 4,477 explicitly identified as never reviewed `never_reviewed=1`.

After training-fitted imputation, there were no missing values remaining for the six features shown in Table 1.

**Table 1. Missing values before imputation; all corresponding after counts are zero.**

| Feature | Train before | Test before | After (each split) |
| --- | --- | --- | --- |
| days_since_last_review | 3571 | 906 | 0 |
| reviews_per_month | 3571 | 906 | 0 |
| review_scores_rating | 3571 | 906 | 0 |
| price_clean | 5282 | 1271 | 0 |
| bedrooms | 3645 | 1034 | 0 |
| beds | 5655 | 1404 | 0 |

Table 2 shows substantial differences in missingness between collection batches.

**Table 2. Collection batches and missingness.**

| Scrape date | Listings | Superhost | Beds missing | Price missing |
| --- | --- | --- | --- | --- |
| 2026-06-17 | 18884 | 38.2% | 1.1% | 1.3% |
| 2026-06-28 | 670 | 19.1% | 100.0% | 81.3% |
| 2026-06-29 | 6173 | 8.8% | 100.0% | 93.4% |
| 2026-07-01 | 1 | 100.0% | 0.0% | 0.0% |

Exploratory preprocessing comparisons use five training folds with selected hyperparameters held fixed.

**Table 3. Mean validation macro-F1 for exploratory preprocessing checks, full-feature models.**

| Training-fold check | KNN macro-F1 | KNN Δ | DT macro-F1 | DT Δ |
| --- | --- | --- | --- | --- |
| Main pipeline | 0.7633 | — | 0.7850 | — |
| Drop recency | 0.7574 | -0.0059 | 0.7717 | -0.0133 |
| Drop review frequency | 0.7508 | -0.0125 | 0.7849 | -0.0001 |
| Drop never-reviewed flag | 0.7469 | -0.0165 | 0.7850 | 0.0000 |
| Zero-fill never-reviewed review frequency | 0.7635 | +0.0002 | 0.7849 | -0.0001 |

Zero filling changes only `reviews_per_month` for never-reviewed rows and retains the flag. These are training-fold comparisons, not test-set performance estimates.

Δ = alternative minus main pipeline; negative values indicate lower mean validation macro-F1. Differences are calculated before rounding the displayed scores.

**Table 4. Mean validation macro-F1 changes for additional preprocessing checks.**

| Check | Feature condition | KNN change | DT change |
| --- | --- | --- | --- |
| Drop price, beds and their missingness flags | Full | +0.0068 | 0.0000 |
| Drop price, beds and their missingness flags | No review | -0.0124 | -0.0008 |
| Train on reviewed listings only | Full | -0.0031 | -0.0002 |

Changes are alternative minus baseline, using five training folds with selected hyperparameters held fixed. The reviewed-only check evaluates both training choices on the same reviewed validation rows; its baseline differs from Table 3.

### 3.2. Correlation analysis

Pearson correlation, Spearman correlation, MI, and NMI were calculated for all ten pairs on the common complete-case population (n = 21,251) in Table 5.

**Table 5. Pairwise associations in reviewed listings.**

| Field 1 | Field 2 | Pearson | Spearman | MI (nats) | NMI |
| --- | --- | --- | --- | --- | --- |
| `host_listings_count` | `days_since_last_review` | -0.1130 | -0.1942 | 0.0569 | 0.0278 |
| `host_listings_count` | `reviews_per_month` | -0.0225 | 0.1490 | 0.0314 | 0.0153 |
| `host_listings_count` | `review_scores_rating` | -0.2100 | -0.2597 | 0.0509 | 0.0262 |
| `host_listings_count` | `host_is_superhost` | -0.0687 | 0.0699 | 0.0136 | 0.0111 |
| `days_since_last_review` | `reviews_per_month` | -0.3719 | -0.7766 | 0.4625 | 0.2010 |
| `days_since_last_review` | `review_scores_rating` | -0.0251 | -0.0422 | 0.0624 | 0.0285 |
| `days_since_last_review` | `host_is_superhost` | -0.3367 | -0.4425 | 0.1159 | 0.0785 |
| `reviews_per_month` | `review_scores_rating` | 0.1487 | -0.0148 | 0.1540 | 0.0703 |
| `reviews_per_month` | `host_is_superhost` | 0.3811 | 0.4661 | 0.1208 | 0.0818 |
| `review_scores_rating` | `host_is_superhost` | 0.2393 | 0.2180 | 0.1154 | 0.0844 |

### 3.3. Supervised learning models and evaluation

KNN and DT models were trained on full and no review feature sets. The selected hyperparameter settings and cross-validation performance macro-F1 are listed in Table 6.

**Table 6. Selected models. All selected KNN inputs are scaled.**

| Condition | Model | Setting | CV macro-F1 |
| --- | --- | --- | --- |
| Full | KNN | k = 51 | 0.7633 |
| Full | DT | depth 5, leaf 50 | 0.7850 |
| No review | KNN | k = 31 | 0.6489 |
| No review | DT | depth 12, leaf 20 | 0.6693 |

Tables 7 and 8 present every attempted hyperparameter combination for KNN and DT and show the mean validation macro-F1.

**Table 7. KNN grid: U = unscaled; S = scaled.**

| k | Full U | Full S | No review U | No review S |
| --- | --- | --- | --- | --- |
| 1.0 | 0.6565 | 0.7062 | 0.5981 | 0.6100 |
| 3.0 | 0.6738 | 0.7328 | 0.6121 | 0.6213 |
| 5.0 | 0.6871 | 0.7415 | 0.6228 | 0.6246 |
| 7.0 | 0.6928 | 0.7467 | 0.6254 | 0.6315 |
| 9.0 | 0.6976 | 0.7491 | 0.6302 | 0.6312 |
| 11.0 | 0.7005 | 0.7533 | 0.6285 | 0.6340 |
| 15.0 | 0.7041 | 0.7559 | 0.6311 | 0.6401 |
| 21.0 | 0.7050 | 0.7594 | 0.6339 | 0.6418 |
| 31.0 | 0.7078 | 0.7632 | 0.6295 | 0.6489 |
| 41.0 | 0.7092 | 0.7632 | 0.6314 | 0.6479 |
| 51.0 | 0.7084 | 0.7633 | 0.6253 | 0.6470 |

**Table 8. Decision Tree grids: rows = minimum leaf size; columns = maximum depth.**

**Full**

| Leaf | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 20 | None |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7522 | 0.7527 | 0.7668 | 0.7838 | 0.7796 | 0.7780 | 0.7663 | 0.7607 | 0.7416 | 0.7270 | 0.7212 |
| 5 | 0.7522 | 0.7527 | 0.7668 | 0.7838 | 0.7821 | 0.7812 | 0.7714 | 0.7643 | 0.7487 | 0.7382 | 0.7360 |
| 10 | 0.7522 | 0.7527 | 0.7668 | 0.7838 | 0.7817 | 0.7816 | 0.7762 | 0.7674 | 0.7553 | 0.7534 | 0.7525 |
| 20 | 0.7522 | 0.7527 | 0.7669 | 0.7818 | 0.7805 | 0.7813 | 0.7793 | 0.7737 | 0.7697 | 0.7704 | 0.7704 |
| 50 | 0.7522 | 0.7527 | 0.7690 | 0.7850 | 0.7837 | 0.7809 | 0.7793 | 0.7788 | 0.7787 | 0.7787 | 0.7787 |

**No review**

| Leaf | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 20 | None |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4095 | 0.6669 | 0.6661 | 0.6454 | 0.6561 | 0.6599 | 0.6586 | 0.6570 | 0.6396 | 0.6355 | 0.6121 |
| 5 | 0.4095 | 0.6669 | 0.6661 | 0.6452 | 0.6553 | 0.6599 | 0.6604 | 0.6591 | 0.6499 | 0.6406 | 0.6205 |
| 10 | 0.4095 | 0.6669 | 0.6660 | 0.6452 | 0.6553 | 0.6616 | 0.6662 | 0.6665 | 0.6556 | 0.6471 | 0.6307 |
| 20 | 0.4095 | 0.6669 | 0.6660 | 0.6453 | 0.6542 | 0.6652 | 0.6681 | 0.6693 | 0.6543 | 0.6472 | 0.6367 |
| 50 | 0.4095 | 0.6669 | 0.6690 | 0.6441 | 0.6544 | 0.6615 | 0.6630 | 0.6505 | 0.6440 | 0.6332 | 0.6325 |

Table 9 presents the effect of changing various settings, with other settings held at their chosen values.

**Table 9. Chosen-setting CV macro-F1 minus one-setting-reset CV macro-F1.**

| Condition | Model | Setting | Default → chosen | CV difference |
| --- | --- | --- | --- | --- |
| Full | KNN | n_neighbors | 5 → 51 | 0.0219 |
| Full | KNN | scale | False → True | 0.0549 |
| Full | DT | max_depth | None → 5 | 0.0063 |
| Full | DT | min_samples_leaf | 1 → 50 | 0.0012 |
| No review | KNN | n_neighbors | 5 → 31 | 0.0243 |
| No review | KNN | scale | False → True | 0.0195 |
| No review | DT | max_depth | None → 12 | 0.0326 |
| No review | DT | min_samples_leaf | 1 → 20 | 0.0122 |

**Table 10. Test performance and 95% host-bootstrap intervals.**

| Model/condition | Macro-F1 [95% CI] | F1 gain over 0R | ROC-AUC |
| --- | --- | --- | --- |
| Full/KNN | 0.7818 [0.7616, 0.8020] | 0.3723 | 0.8823 |
| Full/DT | 0.7999 [0.7741, 0.8224] | 0.3904 | 0.8914 |
| No review/KNN | 0.6844 [0.6561, 0.7098] | 0.2748 | 0.7942 |
| No review/DT | 0.7214 [0.6945, 0.7467] | 0.3119 | 0.7981 |
| 0R | 0.4095 [0.3908, 0.4264] | 0.0000 | 0.5000 |

**Table 11. Full-model performance on never-reviewed test listings.**

| Model | Listings | Actual Superhost listings | Accuracy | Superhost recall |
| --- | --- | --- | --- | --- |
| Full KNN | 906 | 48 | 94.6% | 0% |
| Full DT | 906 | 48 | 94.7% | 0% |

**Table 12. Test per-class precision (P), recall (R) and F1; class supports = 3,569 non-Superhost and 1,577 Superhost.**

| Model/condition | Non-SH P | Non-SH R | Non-SH F1 | SH P | SH R | SH F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Full/KNN | 0.8656 | 0.8678 | 0.8667 | 0.6990 | 0.6950 | 0.6970 |
| Full/DT | 0.8789 | 0.8745 | 0.8767 | 0.7191 | 0.7273 | 0.7232 |
| No review/KNN | 0.7977 | 0.8395 | 0.8180 | 0.5878 | 0.5181 | 0.5507 |
| No review/DT | 0.8464 | 0.7873 | 0.8158 | 0.5843 | 0.6766 | 0.6271 |
| 0R | 0.6935 | 1.0000 | 0.8190 | 0.0000 | 0.0000 | 0.0000 |

**Table 13. Paired test macro-F1 differences and 95% host-bootstrap intervals.**

| Paired comparison | Macro-F1 difference | 95% host-bootstrap CI |
| --- | --- | --- |
| Full DT − No review DT | 0.0785 | [0.0537, 0.1038] |
| Full KNN − No review KNN | 0.0974 | [0.0773, 0.1190] |
| Full DT − Full KNN | 0.0181 | [0.0015, 0.0350] |

This exploratory check holds the other settings fixed and does not replace the model selected from the main grid.

**Table 14. Exploratory Full KNN boundary check: mean validation macro-F1.**

| k | Mean validation macro-F1 |
| --- | --- |
| 51 | 0.7633 |
| 71 | 0.7641 |
| 101 | 0.7636 |
| 151 | 0.7608 |
| 201 | 0.7609 |

![First three levels of the selected Full Decision Tree](notebook_outputs/figures/tree_top_levels.png)

**Figure 1. First three levels of the Full Decision Tree, selected with maximum depth 5 and minimum leaf size 50; nodes show split conditions and class proportions.**

### 3.4. Feature selection

**Table 15. Embedded and filter top-three rankings from training data.**

| Rank | Tree feature | Importance | MI feature | MI (nats) |
| --- | --- | --- | --- | --- |
| 1 | reviews_per_month | 0.4243 | days_since_last_review | 0.1567 |
| 2 | review_scores_rating | 0.2451 | reviews_per_month | 0.1332 |
| 3 | days_since_last_review | 0.1008 | review_scores_rating | 0.1185 |

Table 15 shows that both DT and MI rank review frequency, overall review rating, and review recency as the top three features. DT ranks review frequency first, whereas MI ranks recency first.

**Table 16. Additional feature-selection result: value rating.**

| Feature | MI (nats) | MI rank | Tree importance | Tree rank |
| --- | --- | --- | --- | --- |
| review_scores_value | 0.1055 | 7 | 0.0000 | 10 (tied) |

One hard-case result is listing ID `16261169` from host `46846051` which has recorded Superhost target 0, but both full-feature models predict 1. This listing has a review frequency of 4.32 reviews/month at 95.46th percentile among listings with observed review frequency. The listing also has 211 reviews, 38 days since last review, and a rating of 4.95. KNN has a predicted Superhost probability of 0.7647, and DT has a predicted Superhost probability of 0.8205.

## 4. Discussion and Interpretation

### 4.1. Data preprocessing

Table 3 shows that review recency is a useful feature to add. With all other features retained, dropping recency lowers mean validation macro-F1 for the full DT model from 0.7850 to 0.7717 (change of -0.0133) and for full KNN from 0.7633 to 0.7574 (change of -0.0059).

Keeping never-reviewed listings means the data includes listings both with and without review histories.
Dropping them would restrict the population to reviewed listings.
However, Table 11 shows that among the 906 never-reviewed test listings, there were 48 listings by Superhosts, and both full KNN and full DT had Superhost recall of 0%.
An unreviewed listing can belong to a Superhost because Superhost status belongs to the host, while review history belongs to the individual listing.
A Superhost may have a new, unreviewed listing alongside other reviewed listings that contributed to their Superhost status. This explains how the combination is possible, rather than why the models missed these listings.

Since the vast majority of the never-reviewed listings did not have Superhost status (Table 11), KNN had an accuracy of 94.6% and DT 94.7% for the never-reviewed subgroup. This masked the models' failure to identify any Superhosts in this subgroup.

Table 3 shows that removing the never-reviewed flag changes KNN mean validation macro-F1 by -0.0165 and DT by 0.0000.
This may be because KNN uses the never-reviewed flag to distinguish listings with median-filled review values from those with genuine observed values.
Removing the flag changes those distances and potentially the neighbours selected.

In Table 3, when we used zero imputed review frequency (as opposed to median imputed), there were very small changes in full-model CV mean validation macro-F1: KNN +0.0002 and DT -0.0001.

From Table 2, we observed that collection batches with scrape date 2026-06-28 and 2026-06-29 had a large increase in beds missing (1.1% to 100%) and price missing (1.3% to 81.3% and 93.4%) when compared to the 2026-06-17 batch.
This suggests that the beds and price missingness is associated with the collection batches, not just host behaviour of not including those data.
However, as per Table 4, dropping price, beds and their flags slightly increases mean validation macro-F1 in Full feature condition by +0.0068 KNN (and 0.0000 DT) while it lowers slightly in No review condition for both KNN and DT models (-0.0124 KNN, -0.0008 DT).
This suggests that there is still some predictive information encoded in those fields in the No Review model despite the apparent association with collection batch/scrape date when there are no review features available to the models.
However, with review features available to the models, those fields may introduce unhelpful information into KNN's distance calculation.

Table 4 shows that when we did reviewed-only training and evaluated the models on the same reviewed validation rows, we see a CV change -0.0031 KNN, -0.0002 DT relative to training on all listings.
As such, there is no evidence that excluding never-reviewed training rows improves performance on reviewed validation listings.

### 4.2. Correlation analysis
Table 5 shows that there is a strong negative rank correlation between recency and frequency with Spearman -0.7766, but a moderate linear correlation with Pearson -0.3719.
The difference in Spearman and Pearson correlations suggests a possible nonlinear structure or skewed variable distributions, but we won't know without further analysis.
The modelling implication is that there may be overlapping information between recency and frequency, so we may be able to drop one feature without too much impact.

> **FOR ENOCH ONLY: data-grounded explanation to add after a targeted diagnostic.**

- For the recency/frequency Pearson–Spearman gap (-0.3719 versus -0.7766), inspect quantiles and a joint plot on the same reviewed population; identify a measured tail, skew or curved relationship that could account for the gap.
- Add the diagnostic result to Section 3.2 before interpreting it here. This check has not yet been run; do not replace the cautious explanation with an asserted cause.
- Explain the downstream decision using the existing removal checks: strong association motivated checking redundancy, but unequal losses mean both features were retained. Avoid claiming correlation alone established interchangeability.

As per Table 3, removing either feature has different effects on the models.
Mean validation macro-F1 decreased for KNN when either feature was removed, with a larger loss for frequency.
On the other hand, DT showed a decrease when recency was removed but showed little mean change when frequency was removed.
Therefore, their strong rank correlation did not imply interchangeable predictive contributions.

Table 5 shows that, in terms of predictor-target associations, review frequency, recency, and rating have stronger association with Superhost status than listing count.
Rating has smaller absolute Pearson and Spearman coefficient than recency, but similar MI. This suggests that weaker linear or rank correlation does not necessarily mean there is less dependence.

Table 5 also shows that different correlation methodologies don't always agree on correlation direction or strength.
For listing count vs frequency and listing count vs target both had small negative Pearson and positive Spearman.
On the other hand, frequency vs rating had small negative Spearman and positive Pearson.
These show a weak/little linear or monotonic association.
Given their small magnitudes, these sign differences should not be interpreted as evidence of opposing trends.
For host listing count vs Superhost status, both absolute coefficients are below 0.07, suggesting negligible linear and monotonic associations.
MI/NMI describe dependence and are non-negative, and their magnitudes are not directly comparable with correlation coefficients.

### 4.3. Supervised learning models and evaluation

Table 10 showed a larger decrease in macro-F1 from full condition to no review condition for KNN (-0.0974) when compared against DT (-0.0785).
This suggests that, in this comparison, KNN is more dependent on review features than DT.

Table 10 also showed both KNN and DT no review condition models performed better than 0R baseline (predict all non-Superhost), suggesting that non-review predictors still have predictive information.

Table 13 shows that for the full models' test macro-F1, the paired full DT-minus-full KNN interval lies entirely above zero at 95% CI.
This supports choosing Full DT over Full KNN on macro-F1 grounds for this analysis.

Table 10 showed that Full model DT and KNN's ROC-AUC model ranking performance was better than No Review models, which performed better than baseline/0R.
This is consistent with the pattern observed for macro-F1.

Table 12 showed that full DT has both Superhost recall and precision increase over Full KNN.
That means full DT identifies a greater proportion of actual Superhosts, and a greater proportion of its Superhost predictions are correct.
Without review features, DT (0.6766) has much higher Superhosts recall than KNN (0.5181) while maintaining similar precision (0.5843 DT vs 0.5878 KNN).
In this condition, DT missed fewer Superhosts than KNN.
However, even full DT still had substantial prediction errors, missing 0.2727 of actual Superhosts, and 0.2809 of its Superhost predictions were incorrect.

> **FOR ENOCH ONLY: additional interpretation outline. Preserve the prose above when incorporating these points.**

- Performance, baseline and RQ interpretation:
  - Full KNN and DT exceed 0R by +0.3723 and +0.3904 test macro-F1 respectively (Table 10); useful discrimination beyond predicting every listing as non-Superhost.
  - Removing review inputs reduces macro-F1 by 0.0974 KNN and 0.0785 DT, while No review remains above 0R (Tables 10/13); non-review attributes retain information, but review inputs contribute substantially in these fitted models.
  - The paired Full DT-minus-Full KNN difference is +0.0181, with 95% interval [0.0015, 0.0350] (Table 13); supports a qualified preference for DT in this analysis, conditional on this release and fitted models.
- Class-specific errors:
  - No review DT recall is 0.6766 versus KNN 0.5181 at similar precision (0.5843 versus 0.5878): DT finds more actual Superhosts without a corresponding improvement in precision (Table 12).
  - Even Full DT misses 27.27% of actual Superhosts and 28.09% of its Superhost predictions are incorrect (Table 12); higher macro-F1 does not eliminate meaningful errors.

- Feature influence:
  - KNN uses distances across all standardised predictors; changing a feature can change distances and the neighbours selected. **Method source:** Section 2.3.2, StandardScaler and Euclidean-distance KNN description.
  - Existing Full KNN training-fold checks: removing recency changes mean validation macro-F1 by -0.0059; removing review frequency changes it by -0.0125. **Evidence:** Table 3, Drop recency and Drop review frequency rows, KNN Δ column.
  - Compare with DT importance rankings in Sections 3.4/4.4: review frequency, rating and recency are the tree's top three features. **Evidence:** Table 15, ranks 1–3, Tree feature and Importance columns (0.4243, 0.2451, 0.1008).
- Model behaviour:
  - Full scaled KNN's validation curve plateaus: k = 31/41 gives 0.7632 versus 0.7633 at k = 51 (Table 7); 51 is the numerical maximum, not an established materially superior setting.
  - Add training/validation evidence to Section 3.3, then explain the reduced gap: Full scaled KNN at k = 1 has training macro-F1 1.0000 versus validation 0.7062; k = 51 has 0.7992 versus 0.7633. Supports discussing less overfitting with more neighbours, without attributing all errors to variance.
  - Report fold SD alongside the plateau: 0.0095 at k = 31 versus 0.0108 at k = 51; their mean difference is about 0.0001. SD describes fold variation and is not a paired significance test. Source: `notebook_outputs/tables/cv_results_knn.csv`.
  - Add the tree training/validation comparison to Section 3.3: Full unbounded depth/leaf 1 gives 1.0000/0.7212; selected depth 5/leaf 50 gives 0.7996/0.7850. Supports regularisation as a response to the observed training/validation gap. Source: `notebook_outputs/tables/cv_results_tree.csv`.
  - The exploratory Full KNN boundary check gains only 0.0008 at k = 71 versus 51, then declines at larger k; little evidence of a material benefit from extending the main grid. **Evidence:** Table 14, k = 51, 71, 101, 151 and 201 rows, Mean validation macro-F1 column.
  - Table 9 shows that scaling and regularisation effects vary across feature conditions: scaling changes CV by +0.0549 Full versus +0.0195 No review; limiting tree depth changes CV by +0.0063 Full versus +0.0326 No review. **Evidence:** Table 9, Full/KNN/scale, No review/KNN/scale, Full/DT/max_depth and No review/DT/max_depth rows, CV difference column.
  - Regularised tree: depth 5, leaf 50; readable rules, with interactions conditional on earlier splits. **Evidence:** Table 6, Full/DT row, Setting column; Figure 1 in Section 3.3 for rule structure.
  - Table 8 shows that increasing tree depth beyond the selected depth does not improve mean validation macro-F1 when minimum leaf size is held at its selected value. **Evidence:** Table 8, Full panel, Leaf = 50 row: depth 5 gives 0.7850; depth 6, 8, 10, 12, 15, 20 and None give lower scores. Selected setting: Table 6, Full/DT row.

### 4.4. Feature selection

- Same top-three set, different order: **Evidence:** Table 15, ranks 1–3, Tree feature and MI feature columns.
  - MI measures each variable separately; tree importance depends on splits after other variables have already partitioned the data. **Method source:** Section 2.4, filter MI and embedded impurity-importance definitions.
  - Recency/frequency Spearman -0.7766 provides a concrete reason they may substitute for one another. **Evidence:** Table 5, days_since_last_review / reviews_per_month row, Spearman column; possible substitution is an interpretation, not a demonstrated equivalence.
- Specific disagreement outside the top three:
  - Value rating: MI 0.1055, rank 7; tree importance 0.0000. **Evidence:** Table 16, review_scores_value row, MI, MI rank and Tree importance columns.
  - This is an observed instance of opposite rankings: MI scores value rating on its own, while the fitted tree assigns it no splits. Zero tree importance does not mean zero predictive information (Table 16).
  - Possible explanation: overlap with overall rating already used by the tree (importance 0.2451, Table 15). Before claiming demonstrated redundancy, measure rating/value-rating association on training data and add it to Section 3.4; that check remains pending. Otherwise retain overlap as a hypothesis, not an established cause.
- Hard case:
  - Listing 16261169 has high frequency and rating yet recorded target 0; both models overpredict. **Evidence:** Section 3.4 hard-case paragraph: frequency 4.32 reviews/month, percentile 95.46, rating 4.95, target 0, both predictions 1.
  - Strengthen the explanation with a targeted model inspection, still to do:
    - Trace this listing through the fitted Full tree; record its feature values, thresholds and terminal-node class proportion, then explain the prediction of 1 (probability 0.8205).
    - Inspect its 51 neighbours under the fitted Full KNN transformation; report the Superhost vote underlying probability 0.7647 and relevant similarities, rather than treating high review frequency alone as the model's rule.
    - Compare 4.32 reviews/month with observed median and tail cut-offs on the stated population. The 95.46th percentile establishes a tail case; do not call it an extreme outlier without showing how far its value lies from typical values.
    - Put new measured values in Section 3.4 before interpreting them here. These inspections have not yet been performed.
  - Unobserved eligibility details, timing differences or other host conditions are possible explanations, not established facts. **Basis:** the same Section 3.4 case; Section 2.3.2 evaluation-scope point.

> **FOR ENOCH ONLY:** Discussion is the main place for interpretation. Select the strongest evidence rather than repeating every result. Use Tables 3/4 for sensitivities, Table 11 for the never-reviewed subgroup, and Figure 1 for the tree structure. All evidence cited by the discussion must remain in the main report after layout cuts.

## 5. Limitations and Improvement Opportunities

### 5.1. Preprocessing

- Missing histories remain a blind spot: 4,477 never-reviewed listings are retained, but both Full models miss all 48 subgroup positives (Table 11); median-filled review measures cannot restore absent information.
  - Improvement: compare additional missing-history representations within training folds; assess subgroup recall as well as overall macro-F1.
- Collection dependence: beds missingness rises from 1.1% to 100% across batches (Table 2). Joint price/bed/flag removal gives +0.0068 Full KNN versus -0.0124 No review KNN (Table 4), but cannot separate observed values, imputation and missingness signals.
  - Improvement: audit collection procedures and evaluate a later batch; separately test value and indicator contributions within training data. Future-batch performance remains unmeasured.

### 5.2. Correlation

- Coverage and dependence: Table 5 covers 21,251 reviewed listings, excluding 4,477 never-reviewed rows; the full data also contain 25,728 listings from 14,113 hosts, so observations are not independent.
  - Improvement: compare host-level summaries, explicitly stating how the population and unit of analysis change.
- Representation: ten intended rank intervals yield only 8 host-count bins and 9 rating bins (Section 2.2); MI/NMI depend on this choice.
  - Improvement: compare a small set of alternative bin counts on training data. Binning sensitivity has not been measured, and these associations do not establish causation.

### 5.3. Supervised learning

- Scope: one release and one holdout of 2,821 hosts; the release was examined during the earlier trial (Section 2.3.2). The +0.0181 DT-over-KNN difference and its [0.0015, 0.0350] interval are conditional on these fitted models (Table 13), rather than estimates of training/tuning variation.
  - Improvement: a genuinely new temporal test and nested grouped CV with fresh tuning in each outer fold would test temporal transfer and training/split sensitivity respectively.

### 5.4. Feature selection

- Shared predictors complicate ranking: recency/frequency Spearman is -0.7766 (Table 5), while value rating has MI 0.1055 but zero tree importance (Table 16). Tree splits and MI's discrete/continuous treatment can affect rankings; instability and estimator bias were not measured here.
  - Improvement: compare rankings across training resamples and use permutation importance with correlated predictors assessed jointly; report whether the shared top-three set persists.
- One selected tail case, listing 16261169 at the 95.46th review-frequency percentile, demonstrates a model failure rather than its prevalence (Section 3.4).
  - Improvement: evaluate error rates across predefined review-frequency ranges, including their sample sizes, rather than generalising from a selected misclassification.

## 6. Conclusion

- Direct answer in two or three sentences written by the team:
  - Predictability: Full KNN macro-F1 0.7818 and Full DT 0.7999, versus 0R 0.4095, on listings belonging to held-out hosts. **Evidence:** Table 10, Full/KNN, Full/DT and 0R rows, Macro-F1 column; Section 2.3.2 for host separation.
  - Most influential features: review frequency, overall rating and review recency are the shared top-three set; tree ranks frequency first, MI ranks recency first. **Evidence:** Table 15, ranks 1–3, Tree feature and MI feature columns.
  - Full DT macro-F1 0.7999; No review 0.7214; paired difference 0.0785 [0.0537, 0.1038]. **Evidence:** Table 10, Full/DT and No review/DT rows, Macro-F1 column; Table 13, Full DT − No review DT row, difference and CI columns.
  - Review fields improve discrimination; non-review inputs retain predictive association above 0R (0.4095). **Evidence:** Table 10, Full and No review rows for both models versus 0R, Macro-F1 and ROC-AUC columns; Table 13, Full-minus-No-review comparisons.
  - Bound the claim to recorded status, this release and these models; mention subgroup/collection limitations. **Evidence:** Table 11, both rows, 0% Superhost recall; Table 2, scrape-date missingness differences. Scope: Sections 1 and 2.3.2.

## 7. References

Airbnb. (n.d.). *What’s required to be a Superhost*. Retrieved October 4, 2026, from https://www.airbnb.com/help/article/829

City of Melbourne. (2026). *Small areas for Census of Land Use and Employment (CLUE)* [Data set]. Retrieved October 4, 2026, from [https://data.melbourne.vic.gov.au/explore/dataset/small-areas-for-census-of-land-use-and-employment-clue/](https://data.melbourne.vic.gov.au/explore/dataset/small-areas-for-census-of-land-use-and-employment-clue/)

Inside Airbnb. (2026). Melbourne, Victoria, Australia. Retrieved 3 October 2026 from http://insideairbnb.com/get-the-data/.

Moritz, H. (2000). Geodetic reference system 1980. *Journal of Geodesy, 74*(1), 128–133. https://doi.org/10.1007/s001900050278

- Course materials for methods, plus relevant implementation documentation actually consulted.

> **FOR ENOCH ONLY:** Format actual consulted sources in APA style. Do not invent a retrieval date. Starting references: [Inside Airbnb](https://insideairbnb.com/get-the-data/), [scikit-learn MI documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html), [grouped cross-validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [Airbnb eligibility context](https://www.airbnb.com/help/article/829). These links are source pointers, not completed bibliography entries.

## Working checks, not part of the report

> **FOR ENOCH ONLY: before submission.** Assign an author and second reader to every section. Each member should explain the grouping, imputation, paired bootstrap, feature rankings and hard case. Write the prose yourself; check page count after inserting tables in the template. Remove all writing notes, verify sequential table/figure numbers, verify repeated values against the same CSVs, and check any later Canvas amendments. No grade estimate is implied by these writing notes.

## Page-limit cut recommendations, not part of the report

> **FOR ENOCH ONLY:** Apply these only if the formatted report exceeds nine pages excluding references. These are recommendations awaiting review, not cuts already approved. When removing evidence, remove its dependent discussion and update methods, captions and references together. Keep every required rubric component in the main report.

- Retain Figure 1 (the tree) while it supports actual rule or hard-case interpretation; review removal only if its dependent interpretation is removed too. The former interval and feature-ranking figures have been removed because Tables 10/13 and 15/16 carry their evidence.
- Consider removing Table 14 and its boundary-check discussion together; the main-grid tuning evidence remains in Tables 7 and 8.
- Consider removing the reviewed-only training row of Table 4 and its Section 4.1 paragraph together. It is currently used in prose.
- Section 5 has been shortened to strong limitations and linked improvements. Further compression must retain preprocessing, correlation, modelling and feature-selection coverage.
- Remove all personal callouts, evidence-writing notes and this cut list before submission; shorten the conclusion to the direct answer, leading features and principal qualification. Preserve numeric evidence and report-table citations in the final prose.

- Consider removing No review if space is still insufficient: retain Full KNN, Full DT and 0R as the required comparison. Remove the No review definition, tuning tables/columns, metric rows, paired Full-minus-No-review differences, figure condition and dependent discussion/conclusion together. Retain all required tuning scores, baseline gains, uncertainty, model interpretation and feature-selection coverage for Full. This is a further scope option awaiting review.
