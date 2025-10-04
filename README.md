# `> K E P L E R // a vibe check`

![GitHub last commit](https://img.shields.io/github/last-commit/dattanirjhar/planets-only?display_timestamp=committer&logoColor=violet)
![GitHub Created At](https://img.shields.io/github/created-at/dattanirjhar/planets-only)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/w/dattanirjhar/planets-only/main)
![NASA](https://img.shields.io/badge/SpaceApps-000000?logo=nasa)
![Planets Only!](https://img.shields.io/badge/planets%20only!-8A2BE2)
![KEPLER](https://img.shields.io/badge/KEPLER-00B496)
![GitHub License](https://img.shields.io/github/license/dattanirjhar/planets-only)

```
============================================================================
|                                                                          |
|   [BOOTING SYSTEM...]                                                    |
|   NASA-ARCHIVE: EXOPLANET_SURVEILLANCE_PROTOCOL//v2.7                   |
|   > Access granted.                                                      |
|   > Welcome, Operator.                                                   |
|   > Initiating KEPLER CLASSIFIER CORE...                                |
|                                                                          |
|        .  *           .       .   .                                      |
|              .     .               *         .                           |
|         .            .   .            *                                  |
|   .          *            .    .              .                          |
|         *        .                  *                                    |
|                                                                          |
============================================================================
```

Yo. So you found **Project KEPLER**.
This isn't your average data science gig.
We're sifting through NASA's deep-space archives — hunting for the real exoplanets hiding among the noise, lies, and cosmic impostors.

This is **space forensics**.
We're here to **expose the fakes** and **crown the real ones**.

But here's the twist: this project became a story about *hubris, humility, and redemption*.
What started as a simple classifier turned into a three-act journey through the harsh realities of machine learning in the wild.

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
> 
> [TRAINING COMPLETE]
> FINAL_ACCURACY: 94.2%
> 
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
+ NEW OBJECTIVE: Build a model that doesn't cheat.
+ MISSION PARAMETERS: No flags. No shortcuts. Pure physics.
+ GOAL: Unified intelligence across Kepler + TESS.
```

</div>

The failure taught us what we actually needed:  
A model that could generalize. A model that learned the *fundamentals*, not the quirks.

### `// Phase 1: Level the Playing Field`

We created **`unified_dataset.csv`**.

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

### `// Phase 2: Make It Smarter`

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

### `// The Final Form`

```
============================================================================
  UNIFIED_MODEL_SMOTE_v3.py
  --------------------------------------------------------------------------
  [x] No cheat codes
  [x] Learns from pure signal physics
  [x] Performs fairly across Kepler AND TESS missions
  [x] Tuned for performance
  [x] Balanced for justice

  STATUS: Not just accurate. WISE.
============================================================================
```

---

## >> THE BLUEPRINT // File Structure

### `// Act I: The Kepler Saga`
```
analysis_v2.py                       | EDA. Data forensics. First contact.
feature_importance_v1.py             | Exposes the cheat codes.
confirmed_vs_candidate_v1.py         | Learns the physics of confirmation.
multiclass_model_v1.py               | The OG champion (before the fall).
multiclass_gradient_boosting_v1.py   | Gradient Boosting enters. 94.2%.
multiclass_xgboost_v1.py             | XGBoost. Also 94.2%. Rival king.
```

### `// Act II: The Humbling`
```
translate_toi_data_v2.py             | Rosetta Stone for TESS + Kepler.
test_rf_on_toi.py                    | The stress test. The crash.
```

### `// Act III: Redemption`
```
unified_dataset.csv                  | The balanced, no-cheat-codes dataset.
unified_model_v1.py                  | First attempt. 73%. Learning.
unified_model_tuned_v2.py            | GridSearchCV upgrade. 74%. Smarter.
unified_model_smote_v3.py            | SMOTE-balanced. Fair. Wise.
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
A 74% model that's balanced and generalizes is better than a 94% specialist that only works in one scenario.

### `> Cross-Mission Validation is Critical`
If your model can't handle data from a related but different source, it's not production-ready.  
TESS wasn't just a test — it was a reality check.

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
```diff
+ koi_model_snr           -> Signal-to-noise ratio. The real MVP.
+ koi_depth               -> Transit depth
+ koi_duration            -> Transit duration
+ koi_period              -> Orbital period
+ koi_prad                -> Planetary radius estimate
+ *_err columns           -> Measurement uncertainty. Confidence indicators.

- koi_fpflag_*            -> REMOVED from unified models. Learned our lesson.
```

### `> Model Evolution`

<div align="center">

| Model | Accuracy | Method | Generalization |
|-------|----------|--------|----------------|
| Random Forest (Act I) | 92.6% | Specialist | Poor |
| Gradient Boosting (Act I) | 94.2% | Flag-dependent | Poor |
| XGBoost (Act I) | 94.2% | Flag-dependent | Poor |
| Unified Baseline (Act III) | 73% | No flags | Moderate |
| Unified Tuned (Act III) | 74% | + GridSearchCV | Good |
| Unified SMOTE (Act III) | 74% | + Balancing | **Excellent** |

</div>

### `> Validation Strategy`
```
Segmented validation
  -> Separate scoring for Kepler vs. TESS to detect bias
Cross-mission testing
  -> Train on one mission, validate on another
SMOTE balancing
  -> Synthetic oversampling for minority classes
```

---

## >> DATA ORIGINS // Source Material

**Kepler KOI Dataset**  
```
SOURCE: NASA Exoplanet Archive
SIZE:   Thousands of Kepler Objects of Interest
STATUS: Cleaned, imputed, cheat codes stripped for unified training
```

**TESS TOI Dataset**  
```
SOURCE: NASA Transiting Exoplanet Survey Satellite
SIZE:   TESS Objects of Interest
STATUS: Re-aligned to Kepler schema via translate_toi_data_v2.py
        Missing columns interpolated or zero-padded
```

**Unified Dataset**  
```
SOURCE: Combined Kepler + TESS
SIZE:   Cross-mission merged data
STATUS: No koi_fpflag_* columns. Pure signal-based features only.
```

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

**Full license text:**  
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

**TL;DR version:**  
[choosealicense.com/licenses/gpl-3.0](https://choosealicense.com/licenses/gpl-3.0/)

---

## >> STATUS REPORT // End of Transmission

<div align="center">

```
----------------------------------------------------------------------
  THREE-ACT STRUCTURE: COMPLETE
  HUBRIS ACKNOWLEDGED
  HUMILITY INTEGRATED
  REDEMPTION ACHIEVED

  Current Model State:
    -> Accuracy: ~74% (unified, cross-mission validated)
    -> Bias: Minimized via SMOTE
    -> Generalization: Confirmed across Kepler + TESS
    -> Intelligence: Learning physics, not flags

  [ALL SYSTEMS NOMINAL]
  [WAITING FOR NEXT TRANSMISSION...]
----------------------------------------------------------------------
```

</div>

---

<div align="center">

```
======================================================================
                     End of Line // KEPLER OUT
          This wasn't just a project. It was a journey.
                      Transmission Complete
======================================================================
```

</div>