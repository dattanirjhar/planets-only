# `> K E P L E R // a vibe check`

![NASA](https://img.shields.io/badge/SpaceApps-000000?style=for-the-badge&logo=nasa)
![Planets Only!](https://img.shields.io/badge/planets%20only!-FFFFFF?style=for-the-badge&logo=saturn&logoColor=black)
![KEPLER](https://img.shields.io/badge/KEPLER-00B496?style=for-the-badge)
![EXOPLANETS](https://img.shields.io/badge/EXOPLANETS-EB5559?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/dattanirjhar/planets-only?display_timestamp=committer&style=for-the-badge&logo=github)
![GitHub Created At](https://img.shields.io/github/created-at/dattanirjhar/planets-only?style=for-the-badge&logo=github)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/w/dattanirjhar/planets-only/main?style=for-the-badge&logo=github)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-learn?style=for-the-badge&color=yellow&logo=python&logoColor=lime)
![GitHub License](https://img.shields.io/github/license/dattanirjhar/planets-only?style=for-the-badge&color=pink)


```
============================================================================
|   [ BOOTING SYSTEM... ]                                                  |
|   NASA-ARCHIVE: EXOPLANET_SURVEILLANCE_PROTOCOL // v2.7                  |
|   > Access granted.                                                      |
|   > Welcome, Operator.                                                   |
|   > Initiating KEPLER CLASSIFIER CORE...                                 |
|        .  *           .       .   .                                      |
|              .     .               *         .                           |
|         .            .   .            *                                  |
|   .          *            .    .              .                          |
|         *        .                  *                                    |
============================================================================
```

Yo. So you found **Project KEPLER**.  
This isn’t your average data science gig.  
We’re sifting through NASA’s deep-space archives — hunting for the real exoplanets hiding among the noise, lies, and cosmic impostors.

This is **space forensics**.  
We’re here to **expose the fakes** and **crown the real ones**.

---

## >> ACT I // THE KEPLER SAGA
<div align="center">

```diff
! MISSION BRIEF: Learn the game. Dominate the data.
! OBJECTIVE: 94% accuracy. Make it look easy.
```

</div>

### `// The Strategy: Divide and Conquer`

We didn't just build *one* model. We built a **pipeline**.

**Step 1: Weed the Garden**
```
> INITIALIZING FAKE_DETECTOR_v1.0...
> LOADING FALSE_POSITIVE_CLASSIFIER...
> [====================] 100%
> ACCURACY: 92.6%
> STATUS: OPERATIONAL
```

First model? A specialist. Its *only* job was to spot the **False Positives** — the cosmic fakes trying to pass as planets.
It crushed this. `92.6%` accuracy right out the gate.

Then we looked at the feature importance scores.
The model had found **cheat codes**.

The Kepler dataset came pre-loaded with `koi_fpflag_*` columns — literally labeled flags that screamed *"this one's fake!"*
Our model learned to read the answer key.
Smart? Yes.
*Too* smart? Also yes.

---

**Step 2: Crown the Real Ones**
```
> INITIALIZING CONFIRMATION_PROTOCOL...
> SEPARATING [CONFIRMED] FROM [CANDIDATE]...
> LEARNING PHYSICS_BASED_FEATURES...
> [====================] 100%
> STATUS: NO CHEAT CODES DETECTED
```

With the fakes banished, we trained a second model to separate **Confirmed Planets** from the **Candidates** still waiting for their glow-up.

This time, no cheat codes.  
The model had to learn *actual physics*.

<div align="center">

| Feature | Importance | Meaning |
|---------|-----------|---------|
| `koi_model_snr` | **HIGH** | Signal-to-noise ratio |
| `*_err` columns | **HIGH** | Measurement uncertainty |
| Other metrics | MODERATE | Supporting evidence |

</div>

Translation?
Confirmation isn't about *finding* a signal — it's about having a signal so clean, so precise, that there's zero room for doubt.
The universe doesn't do "probably." It does **"definitely."**

---

**Step 3: The Final Boss**
```
> MERGING MODELS...
> GRADIENT_BOOSTING.load()
> XGBOOST.load()
> MULTICLASS_FUSION: ACTIVE
> [TRAINING COMPLETE]
> FINAL_ACCURACY: 94.2%
> ================================ MISSION ACCOMPLISHED
```

We fused everything into a **multi-class mega-model**.  
Gradient Boosting and XGBoost tag-teamed the problem.

Final score?
**94.2% accuracy.**

Mission accomplished.
Victory screen.
Roll credits.

...right?

---

## >> ACT II // THE GREAT HUMBLING

<div align="center">

```diff
- NEW DATASET DETECTED: TOI_dataset.csv
- SOURCE: TESS Mission
- RUNNING COMPATIBILITY CHECK...
- [ERROR] CATASTROPHIC FAILURE DETECTED
```

</div>

We got cocky.

So we asked the hard question:
*"Is our 94% champion a real planet hunter, or just a Kepler specialist?"*

Enter **TESS** — NASA's next-gen exoplanet telescope.
Different mission. Different data. Same universe.

First problem?
The datasets spoke **different languages**.
Column names didn't match. Labels were inconsistent. Data types? Chaos.

We built `translate_toi_data_v2.py` to bridge the gap — a Rosetta Stone for space data.

---

### `// The Stress Test`

We fed our Kepler-trained model the TESS data.

```
> LOADING TESS_DATA...
> APPLYING KEPLER_TRAINED_MODEL...
> [WARNING] koi_fpflag_* COLUMNS NOT FOUND
> [WARNING] PRIMARY FEATURES MISSING
> [ERROR] MODEL COHERENCE DEGRADING
> ACCURACY: 62.3%
> ================================ CRITICAL FAILURE
```

**Result?**
Accuracy collapsed to **~62%**.

The model *panicked*.
Why?
TESS didn't have the `koi_fpflag_*` columns.

Without its cheat codes, our champion **fell apart**.  
It couldn't spot fakes anymore, so it just... labeled everything a *Candidate* and hoped for the best.

<div align="center">

```
--------------------------------------------
   LESSON LEARNED:
   High accuracy != intelligence.
   Our model wasn't learning
   astrophysics. It was just
   really good at reading flags.
--------------------------------------------
```

</div>

This wasn't a failure.  
It was the **most important success** of the entire project.  
Because now we knew what we had to fix.

---

## >> ACT III // REDEMPTION ARC
<div align="center">

```diff
+ NEW OBJECTIVE: Build models that don't cheat.
+ MISSION PARAMETERS: No flags. No shortcuts. Pure physics.
+ GOAL: Unified intelligence across multiple missions.
+ STRATEGY: Parallel development -> Final convergence
```

</div>

The failure taught us what we actually needed:  
Models that could generalize. Models that learned the *fundamentals*, not the quirks.

We split into **two parallel tracks** — TESS and K2 — each following the redemption playbook independently.
Then we'd merge everything into one ultimate unified classifier.

---

### `// PARALLEL TRACK 1: TESS Redemption`

#### **Phase 1: Level the Playing Field**

We created **`unified_dataset.csv`** for TESS.

The move?  
We **stripped the cheat codes** from Kepler entirely.  
Deleted the `koi_fpflag_*` columns.  
Forced the model to learn from the same raw signals that TESS provided.

```
> INITIALIZING DATA_UNIFICATION_PROTOCOL...
> REMOVING koi_fpflag_* COLUMNS...
> ALIGNING KEPLER + TESS SCHEMAS...
> APPLYING SURGICAL_CLEANING_v2...
>
> [====================] 100%
>
> unified_dataset.csv CREATED
> CHEAT_CODES_PRESENT: FALSE
> READY FOR TRAINING
```

This wasn't easy.
Our first cleaning scripts were too aggressive — they nuked entire dataframes.
Classic real-world data science nightmare.

We fixed it with a surgical, two-step approach:
1. Targeted drops for junk columns.
2. Strategic fills for salvageable missing data.

---

#### **Phase 2: Make It Smarter**

<div align="center">

| Iteration | Method | Accuracy | Status |
|-----------|--------|----------|--------|
| **v1** | Baseline unified model | ~73% | Learning |
| **v2** | + GridSearchCV tuning | ~74% | Smarter |
| **v3** | + SMOTE balancing | ~74% | **Fair & Wise** |

</div>

**Attempt 1: The Unified Model**
```
> unified_model_v1.py EXECUTING...
> ACCURACY: 73.1%
> BIAS_DETECTED: Kepler-favored
> STATUS: LEARNING
```

First try? `~73%` accuracy.
Better than the crash-and-burn, but still biased toward Kepler data.

**Attempt 2: Hyperparameter Tuning**
```
> unified_model_tuned_v2.py EXECUTING...
> APPLYING GridSearchCV...
> OPTIMIZING: max_depth, n_estimators, learning_rate
> ACCURACY: 74.2%
> BIAS_DETECTED: Reduced but present
> STATUS: SMARTER
```

We brought in **GridSearchCV** to optimize the model's brain.
Result? `~74%` accuracy.
Smarter, but still biased.

**Attempt 3: SMOTE (The Game-Changer)**
```
> unified_model_smote_v3.py EXECUTING...
> APPLYING SMOTE BALANCING...
> GENERATING SYNTHETIC_MINORITY_SAMPLES...
>
> TRAINING ON BALANCED DATASET...
> [====================] 100%
>
> ACCURACY: 74.3%
> BIAS_DETECTED: MINIMAL
> FAIRNESS_SCORE: ================== OPTIMAL
> STATUS: WISE
```

We used **SMOTE** (Synthetic Minority Over-sampling) to balance the training data.
This didn't just boost accuracy — it made the model *fair*.

The numbers barely moved on paper.
But the *distribution* of intelligence shifted.
The model's ability to correctly classify the tricky minority classes in TESS data **skyrocketed**.

---

### `// PARALLEL TRACK 2: K2 Redemption`

While TESS was being conquered, we launched a **second parallel mission** using the K2 dataset and the NASA Planetary Systems Archive.

The K2 mission was Kepler's second life — after a mechanical failure, it pivoted to new observing strategies.
Different data structure. Different challenges. Same redemption playbook.

#### **Phase 1: Feature Intersection & Surgical Imputation**

```
> INITIALIZING K2_FEATURE_MAP...
> DEFINING GENERALIZED PARAMETER SET...
> DETECTING FEATURE INTERSECTION...
```

The move?  
We defined a **universal feature set** — 21 generalized parameters (like `pl_orbper`, `st_teff`) that existed across both K2 and the Planetary Systems (PS) archive.

But there was a problem.  
K2 had unique transit-specific features (`pl_trandep`, `pl_trandur`) that PS didn't have.

**Solution: Surgical Imputation**
```
> ADDING TRANSIT FEATURES TO PS DATA...
> IMPUTING MISSING VALUES -> 0
> RATIONALE: Zero signal = No transit signal detected
> 
> [====================] 100%
> 
> STATUS: UNIFIED SCHEMA ACHIEVED
> CLASS BALANCE: 4:1 (Not Confirmed : Confirmed)
```

This was a direct application of the lesson from Act II.  
Instead of crashing on missing data, we *imputed strategically* — treating the absence of a transit signal in PS as "zero signal."

The model could now train on both archives without breaking.

---

#### **Phase 2: Establish Fairness (Weighted Random Forest)**

```
> LOADING k2_randomforest.py...
> APPLYING class_weight='balanced'...
> CORRECTING 4:1 CLASS IMBALANCE...
> 
> [====================] 100%
> 
> BASELINE WEIGHTED F1-SCORE: 88.31%
> CONFIRMED RECALL (Minority): 62.77%
> STATUS: FAIR
```

The K2 data had a **brutal 4:1 class imbalance** — way more "Not Confirmed" planets than "Confirmed."

We used a **Weighted Random Forest** with balanced class weights to give the minority class a fighting chance.

<div align="center">

| Metric | Baseline (Weighted RF) |
|--------|------------------------|
| Weighted F1-Score | **88.31%** |
| Overall Accuracy | 88.70% |
| Confirmed Recall | 62.77% |

</div>

This was the *fairness* step.  
The model wasn't just accurate — it was paying attention to the rare confirmed planets.

---

#### **Phase 3: Make It Smarter (Randomized Search Optimization)**

```
> k2_randomforest.py EXECUTING...
> APPLYING RandomizedSearchCV...
> OPTIMIZING: n_estimators, max_depth, min_samples_split...
> 
> [====================] 100%
> 
> OPTIMIZED WEIGHTED F1-SCORE: 89.27%
> CONFIRMED RECALL (Minority): 66.07%
> BIAS_DETECTED: MINIMAL
> STATUS: SMARTER
```

Hyperparameter tuning pushed the K2 model even further.

<div align="center">

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Weighted F1-Score | 88.31% | **89.27%** |
| Overall Accuracy | 88.70% | 89.50% |
| Confirmed Recall | 62.77% | **66.07%** |

</div>

That **3.3 percentage point boost** in Confirmed Recall?  
That's the model getting *smarter* at finding the rare confirmed planets.

---

#### **Phase 4: The Champion (XGBoost on Cleaned Data)**

```
> LOADING k2_xgboost.py...
> APPLYING XGBOOST TO UNIFIED K2 DATA...
> INTENTION: Test performance ceiling
> 
> [PENDING EXECUTION]
> 
> GOAL: Confirm whether 89.27% is the limit or if we can push higher
```

The final step: Apply the **XGBoost champion model** to the perfectly cleaned, unified K2 data.

This will test whether the 89.27% F1-score is the performance ceiling, or if there's more juice to squeeze.


---

## >> ACT IV // THE CONVERGENCE – ONE MODEL TO RULE THEM ALL

<div align="center">

```diff
+ FINAL ACT: The Unification of All Missions
+ OBJECTIVE: Build one universal classifier across Kepler, TESS, and K2
+ STATUS: Achieved. Stable. Deployed.
```

</div>

This final act was the culmination of all our learnings — a systematic and iterative process of building the **definitive, universal classifier**.

---

### `// Phase 1: The Mega-Dataset`

Our foundation began with `create_unified_dataset_v2.py`.  
This script unified **Kepler**, **TESS**, and **K2** datasets — stripping Kepler flags, harmonizing schemas, and merging them into:

```
output/translated_data/mega_unified_dataset.csv
```

It wasn’t smooth.  
Subtle bugs in column intersection logic had to be squashed — an exercise in patience, precision, and perseverance.

```
> INITIALIZING create_unified_dataset_v2.py...
> MERGING [Kepler + TESS + K2]...
> FIXING COLUMN ALIGNMENTS...
> OUTPUT: mega_unified_dataset.csv
> STATUS: CLEAN. READY. MASSIVE.
```

---

### `// Phase 2: The Iterative Climb to a Champion`

Each iteration brought us closer to the final, universal model.

**Attempt 1 — `mega_model.py`**
```
> TRAINING UNIFIED MODEL v1...
> ACCURACY: 73%
> SEGMENTED VALIDATION: Bias detected (Kepler > TESS)
> STATUS: Needs balance
```

**Feature Engineering — `advanced_feature_eng_mega.py`**
We added physics-based features like:

- `pl_rad_to_star_rad_ratio` → Planet–star radius ratio  
- `koi_depth_log` → Log-scaled transit depth  
- `koi_period_scaled` → Normalized orbital period  

This **boosted accuracy** and intelligence.  
The model began learning astrophysics, not dataset quirks.

```
> FEATURE_ENGINEERING: ACTIVATED
> KeyError detected — handled gracefully.
> STATUS: Smarter and more robust.
```

---

**Hyperparameter Tuning — `mega_model_hyperparameter.py`**
```
> ENGINE: XGBoost
> METHOD: RandomizedSearchCV
> PARAMETERS: max_depth, learning_rate, n_estimators
> RESULT: 77.71% ACCURACY
> STATUS: OPTIMAL
```

This was our **champion**.  
The XGBoost engine, finely tuned, achieved **77.71%** accuracy — the highest across all missions.

---

**Balancing Act — `mega_balanced_model.py`**
To fix TESS’s underrepresentation, we applied **SMOTE**.  
While overall accuracy plateaued, recall on the minority TESS classes **soared**.

```
> APPLYING SMOTE...
> RECALL[TESS]: ↑↑↑ Significant improvement
> STATUS: Fair and inclusive
```

---

### `// The Final Decision & Productionalization`

The **final champion** — `mega_model_final.py` — blended **Advanced Feature Engineering** and **Hyperparameter Tuning**.

Then came the operational phase:
```
final_production.py
```
This single pipeline script:
- Processes all three missions (Kepler, TESS, K2)  
- Engineers advanced physical features  
- Trains the final **XGBoost** model  
- Saves it as a reusable `.pkl` file for deployment

```
============================================================================
  FINAL_PRODUCTION.PY
  --------------------------------------------------------------------------
  [x] Unified 3-mission dataset
  [x] Advanced feature engineering
  [x] Tuned XGBoost engine
  [x] Balanced fairness via SMOTE
  [x] Persisted model for reuse

  STATUS: Production-ready. Universal. Endgame.
============================================================================
```

This was more than a conclusion — it was **synthesis**.  
All threads, all lessons, all scars — converging into one definitive artifact:  
**A model that generalizes across missions and learns pure astrophysics.**

---

## >> THE BLUEPRINT // File Structure

### `// Act IV: The Convergence`
```
create_unified_dataset_v2.py        | Merges Kepler + TESS + K2 into one mega-dataset.
advanced_feature_eng_mega.py        | Physics-based feature generation.
mega_model.py                       | First unified attempt (~73%).
mega_model_hyperparameter.py        | RandomizedSearchCV-tuned champion (~77.71%).
mega_balanced_model.py              | SMOTE-enhanced fairness model.
mega_model_final.py                 | Combined feature-engineered + tuned final model.
final_production.py                 | Complete automated training + export pipeline.
```

---

## >> MODEL EVOLUTION // FINAL TIMELINE

<div align="center">

| Phase | Model | Accuracy | Notes |
|-------|--------|----------|-------|
| Act I | Gradient Boosting / XGBoost | 94.2% | Overfit, flag-dependent |
| Act III | Unified SMOTE v3 | 74.3% | Fair, cross-mission |
| **Act IV** | **Mega Model Final** | **77.71%** | Universal, feature-engineered, tuned |
| **Act IV (Balanced)** | **SMOTE-enhanced Mega Model** | **~77%** | Best recall on TESS classes |

</div>

---

## >> STATUS REPORT // End of Transmission

<div align="center">

```
----------------------------------------------------------------------
  FOUR-ACT STRUCTURE: COMPLETE
  HUBRIS ACKNOWLEDGED
  HUMILITY INTEGRATED
  REDEMPTION ACHIEVED
  CONVERGENCE ACCOMPLISHED

  Current Model State:
    -> Accuracy: ~77.7% (Kepler + TESS + K2 unified)
    -> Bias: Negligible
    -> Engine: Tuned XGBoost (RandomizedSearchCV)
    -> Features: Advanced, physics-grounded
    -> Generalization: Confirmed across all missions
    -> Intelligence: Learned astrophysics, not dataset artifacts

  [ALL SYSTEMS NOMINAL]
  [FINAL TRANSMISSION SEALED]
----------------------------------------------------------------------
```

</div>

---

<div align="center">

```
======================================================================
                     End of Line // KEPLER OUT
          This wasn't just a project. It was a journey.
            From hubris to humility — to convergence.
======================================================================
```
</div>
