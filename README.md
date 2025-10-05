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
|   NASA-ARCHIVE: EXOPLANET_SURVEILLANCE_PROTOCOL // v3.0                  |
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
This isn't your average data science gig.
We're sifting through NASA's deep-space archives — hunting for the real exoplanets hiding among the noise, lies, and cosmic impostors.

This is **space forensics**.
We're here to **expose the fakes** and **crown the real ones**.

**Born from the NASA Space Apps Challenge 2024**, this project started as a hackathon submission but evolved into something bigger — a story about *hubris, humility, and redemption*.
What began as a simple classifier turned into a comprehensive journey through the harsh realities of machine learning in the wild.

Let's vibe.

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

**Phase 1: Initial Reconnaissance**
```
> LOADING datasets/KOI_dataset_1.csv...
> EXECUTING draft.py...
> EXECUTING analysis_v2.py...
> 
> [DISCOVERY] Class imbalance detected
> [DISCOVERY] Visual correlation: large radius -> FALSE POSITIVE
> 
> STATUS: LANDSCAPE MAPPED
```

Our first contact with the data revealed two critical insights:
- Massive class imbalance favoring FALSE POSITIVES
- Strong visual correlation between planetary radius and fake signals

**Step 1: Weed the Garden**
```
> INITIALIZING feature_importance_v1.py...
> LOADING FALSE_POSITIVE_CLASSIFIER...
> [====================] 100%
> ACCURACY: 98%
> STATUS: OPERATIONAL
```

First specialist model? Built to hunt fakes. Crushed it with 98% accuracy.

Then we looked at the feature importance scores.
The model had found **cheat codes**.

The Kepler dataset came pre-loaded with `koi_fpflag_*` columns — literally labeled flags that screamed *"this one's fake!"*
Our model learned to read the answer key.
Smart? Yes.
*Too* smart? Also yes.

---

**Step 2: Crown the Real Ones**
```
> INITIALIZING confirmed_vs_candidate_v1.py...
> SEPARATING [CONFIRMED] FROM [CANDIDATE]...
> LEARNING PHYSICS_BASED_FEATURES...
> [====================] 100%
> STATUS: NO CHEAT CODES DETECTED
```

With the fakes banished, we trained a second model to separate **Confirmed Planets** from the **Candidates** still waiting for validation.

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
> EXECUTING multiclass_model_v1.py...
> EXECUTING multiclass_gradient_boosting_v1.py...
> MERGING MODELS...
> XGBOOST SELECTED AS CHAMPION
> 
> [TRAINING COMPLETE]
> FINAL_ACCURACY: 94.2%
> 
> ================================ MISSION ACCOMPLISHED
```

We fused everything into a **multi-class mega-model**.  
XGBoost emerged as the champion.

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
- NEW DATASET DETECTED: datasets/TOI_dataset.csv
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
But our first attempts were too aggressive. Overly zealous `dropna()` calls nuked entire dataframes.
Classic real-world data engineering nightmare.

We learned: surgical cleaning > sledgehammer approaches.

---

### `// The Stress Test`

We fed our Kepler-trained champion the TESS data.

```
> EXECUTING test_rf_on_toi.py...
> LOADING TESS_DATA...
> APPLYING KEPLER_TRAINED_MODEL...
> 
> [WARNING] koi_fpflag_* COLUMNS NOT FOUND
> [WARNING] PRIMARY FEATURES MISSING
> [ERROR] MODEL COHERENCE DEGRADING
> 
> ACCURACY: 62.3%
> 
> ================================ CRITICAL FAILURE
```

**Result?**
Accuracy collapsed from 94.2% to **62.3%**.

The model *panicked*.
Why?
TESS didn't have the `koi_fpflag_*` columns.

Without its cheat codes, our champion **fell apart**.  
It couldn't spot fakes anymore, so it just labeled everything a *Candidate* and hoped for the best.

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
+ GOAL: One unified classifier across ALL missions.
+ STRATEGY: Parallel development -> Final convergence
```

</div>

The failure taught us what we actually needed:  
Models that could generalize. Models that learned the *fundamentals*, not the quirks.

We split into **three parallel tracks** — TESS, K2, and the ultimate convergence — each following the redemption playbook.

---

### `// PARALLEL TRACK 1: K2 Validation`

While rethinking TESS, we launched a **parallel validation mission** using `datasets/K2_Dataset.csv` and the NASA Planetary Systems Archive.

The K2 mission was Kepler's second life after mechanical failure.
Different observation strategy. Different data structure. Same redemption philosophy.

#### **Phase 1: Feature Intersection & Surgical Imputation**

```
> EXECUTING k2_featuremap.py...
> DEFINING GENERALIZED PARAMETER SET...
> DETECTING FEATURE INTERSECTION: 21 universal parameters
> 
> [PROBLEM] K2 has unique transit features (pl_trandep, pl_trandur)
> [PROBLEM] PS Archive missing these columns
> 
> [SOLUTION] Surgical imputation strategy
> -> Adding transit columns to PS data
> -> Imputing missing values as 0
> -> RATIONALE: Zero signal = No transit detected
> 
> [====================] 100%
> 
> STATUS: UNIFIED SCHEMA ACHIEVED
> CLASS BALANCE: 4:1 (Not Confirmed : Confirmed)
```

This was a direct application of Act II's lesson.  
Strategic imputation > aggressive deletion.

---

#### **Phase 2: Establish Fairness**

```
> EXECUTING k2_randomforest.py...
> APPLYING class_weight='balanced'...
> CORRECTING 4:1 CLASS IMBALANCE...
> 
> [====================] 100%
> 
> BASELINE WEIGHTED F1-SCORE: 88.31%
> CONFIRMED RECALL (Minority): 62.77%
> STATUS: FAIR
```

The K2 data had brutal class imbalance — 4:1 ratio against confirmed planets.

**Weighted Random Forest** with balanced class weights gave the minority class a fighting chance.

---

#### **Phase 3: Make It Smarter**

```
> k2_randomforest.py EXECUTING...
> APPLYING RandomizedSearchCV...
> OPTIMIZING: n_estimators, max_depth, min_samples_split...
> 
> [====================] 100%
> 
> OPTIMIZED WEIGHTED F1-SCORE: 89.27%
> CONFIRMED RECALL (Minority): 66.07%
> IMPROVEMENT: +3.3 percentage points minority recall
> STATUS: SMARTER
```

<div align="center">

| Metric | Baseline | Optimized | Gain |
|--------|----------|-----------|------|
| Weighted F1 | 88.31% | **89.27%** | +0.96% |
| Confirmed Recall | 62.77% | **66.07%** | +3.3% |
| Overall Accuracy | 88.70% | 89.50% | +0.8% |

</div>

That 3.3 point boost in minority recall?  
That's the model getting *smarter* at finding rare confirmed planets.

---

#### **Phase 4: The Champion Test**

```
> LOADING k2_xgboost.py...
> INTENTION: Test performance ceiling
> STATUS: PLANNED VALIDATION
```

Final validation: Apply XGBoost to confirm whether 89.27% is the limit.

---

### `// PARALLEL TRACK 2: The Mega-Dataset Convergence`

While K2 proved the philosophy worked, we began the ultimate mission: **unifying everything**.

#### **Phase 1: Build the Foundation**

```
> EXECUTING create_unified_dataset_v2.py...
> LOADING: datasets/KOI_dataset_1.csv (Kepler)
> LOADING: datasets/TOI_dataset.csv (TESS)
> LOADING: datasets/K2_Dataset.csv (K2)
> 
> REMOVING koi_fpflag_* COLUMNS FROM KEPLER...
> TRANSLATING ALL SCHEMAS TO COMMON FORMAT...
> FIXING COLUMN INTERSECTION BUGS...
> APPLYING SURGICAL CLEANING...
> 
> [====================] 100%
> 
> OUTPUT: output/translated_data/mega_unified_dataset.csv
> CHEAT_CODES_PRESENT: FALSE
> MISSIONS_UNIFIED: 3
> STATUS: READY FOR TRAINING
```

This wasn't easy.  
Subtle bugs in column intersection logic had to be hunted down and fixed.
But we persisted. We had a truly unified dataset — no shortcuts, pure physics.

---

#### **Phase 2: The Iterative Climb**

**Attempt 1: Baseline Unified Model**
```
> EXECUTING mega_model.py...
> TRAINING ON UNIFIED DATA...
> APPLYING SEGMENTED VALIDATION...
> 
> ACCURACY: 73%
> BIAS_DETECTED: Kepler-favored performance
> STATUS: LEARNING
```

First try? 73% accuracy.
Better than the 62% crash, but biased toward Kepler data.

---

**Attempt 2: Advanced Feature Engineering**
```
> EXECUTING advanced_feature_eng_mega.py...
> ENGINEERING PHYSICS-BASED FEATURES...
> -> pl_rad_to_star_rad_ratio
> -> Transit signal characteristics
> -> Stellar context ratios
> 
> [OBSTACLE] KeyError encountered
> [FIX] Fault-tolerant feature engineering implemented
> 
> [====================] 100%
> 
> PERFORMANCE BOOST DETECTED
> STATUS: SMARTER DATA
```

We realized the model needed *smarter* data, not just more data.
Physics-based feature engineering gave immediate performance gains.

Key insight: intelligent features > raw features.

---

**Attempt 3: Hyperparameter Optimization**
```
> EXECUTING mega_model_hyperparameter.py...
> APPLYING RandomizedSearchCV...
> COMPUTATIONAL LOAD: HIGH
> OPTIMIZING: XGBoost hyperparameters
> 
> [====================] 100%
> 
> FINAL_ACCURACY: 77.71%
> STATUS: CHAMPION CANDIDATE
```

Hyperparameter tuning pushed us to our best performance yet.

---

**Attempt 4: SMOTE Balancing Experiment**
```
> EXECUTING mega_balanced_model.py...
> APPLYING SMOTE...
> GENERATING SYNTHETIC_MINORITY_SAMPLES...
> 
> [====================] 100%
> 
> OVERALL_ACCURACY: ~77%
> TESS_MINORITY_RECALL: SIGNIFICANT IMPROVEMENT
> 
> STATUS: FAIRNESS ENHANCED
```

SMOTE didn't boost headline accuracy, but it **dramatically improved** minority class recall on TESS data.

The distribution of intelligence shifted where it mattered most.

---

#### **Phase 3: The Final Decision**

After comparing all iterations:

<div align="center">

| Model | Accuracy | Strength | Selected |
|-------|----------|----------|----------|
| Baseline | 73% | Learning foundation | No |
| Feature Engineered | 75%+ | Physics-informed | No |
| Hyperparameter Tuned | **77.71%** | Peak performance | **YES** |
| SMOTE Balanced | 77% | Fairness boost | No |

</div>

**Winner: Advanced Feature Engineering + Hyperparameter Tuning**

The combination of intelligent features and optimized hyperparameters delivered the best balance of:
- High overall performance
- Conceptual clarity
- Robustness across missions

---

### `// Phase 4: Productionalization`

```
> EXECUTING mega_model_final.py...
> LOADING CHAMPION CONFIGURATION...
> 
> PIPELINE:
> [1] Process all raw datasets
> [2] Engineer advanced physics-based features
> [3] Train champion XGBoost model on full unified data
> [4] Save production-ready model to disk
> 
> [====================] 100%
> 
> MODEL SAVED: final_production_model.pkl
> STATUS: PRODUCTION READY
```

The final step: encapsulate the entire winning pipeline.

`final_production.py` is the culmination of everything we learned:
- Processes Kepler, TESS, and K2 raw data
- Engineers advanced features
- Trains the champion XGBoost model on all available data
- Saves a reusable, production-ready model

**One model. Three missions. Pure physics.**

---

## >> THE BLUEPRINT // File Structure

### `// Act I: The Kepler Saga`
```
draft.py                             | Initial EDA. First contact with data.
analysis_v2.py                       | Deep dive analysis. Discovery phase.
feature_importance_v1.py             | False Positive hunter. Found the cheat codes.
confirmed_vs_candidate_v1.py         | Physics-based confirmation classifier.
multiclass_model_v1.py               | First multi-class attempt.
multiclass_gradient_boosting_v1.py   | Gradient Boosting entry. Strong contender.
                                     | XGBoost champion selected. 94.2%.
```

### `// Act II: The Humbling`
```
translate_toi_data_v2.py             | TESS translator. Learned surgical cleaning.
test_rf_on_toi.py                    | The stress test. The crash. The lesson.
```

### `// Act III: Redemption - K2 Track`
```
k2_featuremap.py                     | Feature intersection & surgical imputation.
k2_randomforest.py                   | Weighted RF baseline (88.31% F1).
                                     | Optimized version (89.27% F1).
k2_xgboost.py                        | Champion validation (planned).
```

### `// Act III: Redemption - Mega Convergence`
```
create_unified_dataset_v2.py         | The unifier. Three missions, one dataset.
mega_model.py                        | Baseline unified model (73%).
advanced_feature_eng_mega.py         | Physics-based feature engineering.
mega_model_hyperparameter.py         | Hyperparameter optimization (77.71%).
mega_balanced_model.py               | SMOTE fairness experiment.
mega_model_final.py                  | Champion configuration.
final_production.py                  | Production pipeline. The final form.
```

### `// Datasets`
```
datasets/KOI_dataset_1.csv           | Kepler Objects of Interest
datasets/TOI_dataset.csv             | TESS Objects of Interest
datasets/K2_Dataset.csv              | K2 Mission data
```

### `// Output`
```
output/translated_data/
  mega_unified_dataset.csv           | The unified, no-cheat-codes dataset
  
final_production_model.pkl           | Production-ready trained model
```

All charts, confusion matrices, and visual proof live in `/charts`.  
Open them. Witness the journey.

---

## >> INTEL DROP // What We Learned

<div align="center">

```
----------------------------------------------------------
        MISSION DEBRIEF: CRITICAL INSIGHTS
----------------------------------------------------------
```

</div>

### `> Accuracy != Intelligence`
A model can score 94% and still be fragile.
If it's learned the dataset's quirks instead of the domain's truths, it'll shatter on new data.

### `> Cheat Codes Are a Trap`
The `koi_fpflag_*` columns were helpful — until they weren't.
Real-world deployment means encountering data that doesn't come with an answer key.

### `> Fairness > Raw Performance`
K2's 89% model with 66% minority recall proves this.
A balanced model that generalizes beats a specialist every time.

### `> Cross-Mission Validation is Critical`
If your model can't handle data from a related but different source, it's not production-ready.  
TESS was the stress test. K2 was the validation. The mega-dataset was the proof.

### `> Surgical Imputation > Aggressive Drops`
When datasets don't align perfectly, strategic imputation (like "zero signal = no transit") beats nuking entire columns.
Data engineering matters as much as modeling.

### `> Class Imbalance is a Silent Killer`
K2's 4:1 imbalance nearly sabotaged the mission.
Weighted models and SMOTE are essential tools, not optional extras.

### `> Feature Engineering is Force Multiplier`
Physics-based features delivered immediate gains.
Understanding the domain > throwing raw features at the problem.

### `> Iterative Refinement > Perfect First Try`
Every model iteration taught us something:
- Baseline: revealed bias
- Feature engineering: proved domain knowledge matters
- Hyperparameter tuning: found peak performance
- SMOTE: showed where fairness was needed

The path to greatness is never a straight line.

---

## >> DEV SPECS // Technical Arsenal

### `> Tools & Libraries`
```python
Python 3.11
scikit-learn 1.5
xgboost 2.1
imbalanced-learn          # SMOTE implementation
pandas, numpy
matplotlib, seaborn
tqdm                      # Terminal progress bars
```

### `> Key Features (Post-Cheat-Code Era)`

**Universal Parameters (21 features):**
```diff
+ pl_orbper               -> Orbital period
+ pl_trandep              -> Transit depth
+ pl_trandur              -> Transit duration
+ st_teff                 -> Stellar effective temperature
+ st_rad                  -> Stellar radius
+ st_mass                 -> Stellar mass
+ koi_model_snr           -> Signal-to-noise ratio (MVP)
+ *_err columns           -> Measurement uncertainty

- koi_fpflag_*            -> REMOVED. Learned our lesson.
```

**Advanced Engineered Features:**
```diff
+ pl_rad_to_star_rad_ratio     -> Planetary/stellar radius ratio
+ Transit signal ratios        -> Context-aware signal strength
+ Stellar context features     -> Host star characteristics
```

### `> Model Evolution: The Complete Journey`

<div align="center">

**Act I: Kepler Specialist**

| Model | Accuracy | Method | Generalization |
|-------|----------|--------|----------------|
| False Positive Detector | 98% | Flag-dependent | Poor |
| Confirmed vs Candidate | High | Physics-based | Moderate |
| XGBoost Multi-class | **94.2%** | Flag-dependent | **Poor** |

**Act III: K2 Track**

| Model | Weighted F1 | Confirmed Recall | Method |
|-------|-------------|------------------|--------|
| Baseline Weighted RF | 88.31% | 62.77% | Balanced weights |
| Optimized RF | **89.27%** | **66.07%** | Randomized search |

**Act III: Mega Convergence**

| Model | Accuracy | Method | Status |
|-------|----------|--------|--------|
| Baseline | 73% | No flags, unified | Learning |
| Feature Engineered | 75%+ | Physics features | Smarter |
| Hyperparameter Tuned | **77.71%** | Optimized XGBoost | **Champion** |
| SMOTE Balanced | 77% | Fair distribution | Fairness boost |

</div>

### `> Validation Strategy`
```
Segmented validation
  -> Separate scoring for each mission to detect bias
Cross-mission testing
  -> Train on unified data, validate per mission
Class balancing
  -> Weighted classes (K2) or SMOTE (mega-model experiments)
Feature intersection
  -> Only use features that exist across all missions
Surgical imputation
  -> Strategic zero-fills for missing transit signals
Advanced feature engineering
  -> Physics-based derived features
Hyperparameter optimization
  -> RandomizedSearchCV for XGBoost tuning
```

---

## >> DATA ORIGINS // Source Material

**Kepler KOI Dataset**
```
FILE:   datasets/KOI_dataset_1.csv
SOURCE: NASA Exoplanet Archive
SIZE:   Thousands of Kepler Objects of Interest
STATUS: Cleaned, cheat codes stripped for unified training
```

**TESS TOI Dataset**
```
FILE:   datasets/TOI_dataset.csv
SOURCE: NASA Transiting Exoplanet Survey Satellite
SIZE:   TESS Objects of Interest
STATUS: Translated via translate_toi_data_v2.py
        Schema aligned to unified format
```

**K2 Dataset**
```
FILE:   datasets/K2_Dataset.csv
SOURCE: NASA K2 Mission (Kepler's second life)
SIZE:   K2 confirmed and candidate planets
STATUS: Feature-mapped to generalized parameter set
        Merged with Planetary Systems Archive
```

**Mega Unified Dataset**
```
FILE:   output/translated_data/mega_unified_dataset.csv
SOURCE: Combined Kepler + TESS + K2 missions
SIZE:   Cross-mission merged archive
STATUS: No koi_fpflag_* columns
        Pure signal-based features only
        Advanced physics features engineered
        Production ready
```

---

## >> THE FINAL FORM

```
============================================================================
  final_production.py
  --------------------------------------------------------------------------
  
  THE PRODUCTION PIPELINE:
  
  [1] Load raw data from all three missions
  [2] Apply surgical cleaning and schema unification
  [3] Engineer advanced physics-based features
  [4] Train champion XGBoost model (optimized hyperparameters)
  [5] Save production-ready model to disk
  
  --------------------------------------------------------------------------
  
  CHARACTERISTICS:
  [x] No cheat codes
  [x] Cross-mission validated
  [x] Physics-informed features
  [x] Hyperparameter optimized
  [x] Production ready
  [x] 77.71% accuracy on unified validation
  
  STATUS: COMPLETE. ROBUST. WISE.
  
============================================================================
```

This is it. The culmination of everything we learned.

From naive specialist to robust generalist.
From 94% fragile to 77% robust.
From reading answer keys to learning physics.

**The model that doesn't cheat.**

---

## >> LICENSE // LEGAL UPLINK

<div align="center">

```diff
! GPL v3.0 ACTIVATED
! FREEDOM PROTOCOLS: ENGAGED
```

</div>

This project is **open source** and released under the **GNU General Public License v3.0**.
Translation? It's free. Forever. No corpo BS.

### `> What That Means:`
```
Freedom to Use
  -> Run this code for anything. Science. Homework. Your exoplanet empire.
Freedom to Study
  -> Pop the hood. Read the source. Reverse-engineer the cosmos.
Freedom to Share
  -> Fork it. Upload it. Spread the knowledge like a cosmic virus.
Freedom to Improve
  -> Modify it. Optimize it. Share those improvements back.
```

### `> The One Rule:`
```diff
! If you modify and redistribute this code, your version must also be GPL'd.
! No taking this open-source gift and locking it behind paywalls.
! We keep the code free. Always.
```

### `> No Warranty. No Hand-Holding.`
```
============================================================================
  This software is provided "AS IS"

  Meaning: If it crashes, breaks, or misclassifies Jupiter as a donut,
  that's on you. We're not liable.

  We're just here to vibe and hunt planets.
  Use at your own risk, Operator.
============================================================================
```

### **Full license text:**  
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

### **TL;DR version:**  
[choosealicense.com/licenses/gpl-3.0](https://choosealicense.com/licenses/gpl-3.0/)

---

## >> STATUS REPORT // Mission Complete

<div align="center">

```
----------------------------------------------------------------------
  THREE-ACT STRUCTURE: COMPLETE
  HUBRIS ACKNOWLEDGED
  HUMILITY INTEGRATED
  REDEMPTION ACHIEVED

  KEPLER SPECIALIST (Act I):
    -> Accuracy: 94.2%
    -> Method: Flag-dependent
    -> Generalization: FAILED

  TESS STRESS TEST (Act II):
    -> Accuracy: 62.3% (crash)
    -> Lesson: Cheat codes are traps
    -> Result: Philosophy pivot

  K2 VALIDATION (Act III):
    -> Weighted F1: 89.27%
    -> Confirmed Recall: 66.07%
    -> Proof: Philosophy works

  MEGA UNIFIED MODEL (Act III):
    -> Accuracy: 77.71% (champion)
    -> Method: Physics + optimization
    -> Generalization: CONFIRMED
    -> Status: PRODUCTION READY

  FINAL MODEL:
    -> File: final_production.py
    -> Missions: Kepler + TESS + K2
    -> Features: Physics-based, no shortcuts
    -> Intelligence: Learned fundamentals, not quirks

  [ALL SYSTEMS NOMINAL]
  [MISSION COMPLETE]
  [TRANSMISSION ENDED]
----------------------------------------------------------------------
```

</div>

---

<div align="center">

```
======================================================================
                     End of Line // KEPLER OUT
          
          This wasn't just a project. It was a journey.
          
          From 94% fragile to 77% robust.
          From specialist to generalist.
          From shortcuts to understanding.
          
          The model that doesn't cheat is ready.
          
                      Mission Complete
                   Transmission Terminated
======================================================================
```

</div>