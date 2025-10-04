# `> K E P L E R // a vibe check`

![GitHub last commit](https://img.shields.io/github/last-commit/dattanirjhar/planets-only?display_timestamp=committer&style=for-the-badge&logoColor=violet)
![GitHub Created At](https://img.shields.io/github/created-at/dattanirjhar/planets-only)
![Static Badge](https://img.shields.io/badge/planets%20only!-8A2BE2)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/w/dattanirjhar/planets-only/main)


```
[BOOTING SYSTEM...]
NASA-ARCHIVE: EXOPLANET_SURVEILLANCE_PROTOCOL//v2.7
> Access granted.
> Welcome, Operator.
> Initiating KEPLER CLASSIFIER CORE...
```

Yo. So you found **Project KEPLER**.
This isn’t your average data science gig.
We’re sifting through NASA’s deep-space archives — hunting for the real exoplanets hiding among the noise, lies, and cosmic impostors.

This is **space forensics**.
We're here to **expose the fakes** and **crown the real ones**.
Let’s vibe.

---

## `> the game plan // OPERATION STARSCREEN`

```
>> DECRYPTING MISSION LOGS...
```

### 1. DROP THE FAKES. 🚮  
We started by training a **model to detect False Positives** — the wannabes pretending to be planets.  
It learned their patterns, their offbeat frequencies, their bad cosmic aura.  
If it’s not orbiting right, it’s outta sight.

---

### 2. CROWN THE REAL ONES. 👑  
Once the frauds were dumped into the void, we went hunting for royalty.  
We trained another model to separate **Confirmed Planets** from the **Candidates** still begging for validation.  
This was the cosmic glow-up moment.  

---

### 3. THE FINAL BOSS. 👾  
Then we got bold — we fused it all into one **multi-class model**, the **KEPLER PRIME**.  
We pitted it against the universe itself (and some Gradient Boosting bullies).  
The result? Absolute main character energy.

---

## `> spill the tea // MISSION INTEL`

```
>> ACCESSING COSMIC PATTERN DATABASE...
```

### // how to spot a poser  
The fakes?  
They’re *loud*. They can’t hide their drama.

- **Flagged:** The dataset literally ratted them out — `false_positive_flags`. Easy pickings. 💅  
- **Too Big to Be Real:** Overcompensating? Definitely not a planet. Probably a binary star screaming for attention.  
- **Off-Center:** Their light curves wobble like bad karaoke. Suspicious energy detected.

---

### // the glow up: 'maybe' → 'definitely'  
To turn a “maybe” into a “hell yes,” the numbers gotta *sing*.

- **Clean Signal:** The real ones transit like pros — crisp, stable, zero chaos.  
- **Minimal Error Bars:** Confidence so high it’s basically a mic drop.  
  The universe doesn’t do *“probably”* — it does *“confirmed.”*

---

## `> THE REMIX // RF vs. The Champs`

```
>> INITIATING MODEL BATTLE SEQUENCE...
```

**Round 1:**  
Random Forest enters the arena.  
Accuracy: `92.6%`.  
Solid. Respectable. Slight swagger.

**Round 2:**  
Gradient Boosting + XGBoost show up.  
Accuracy spikes to `94.2%`.  
Cleaner, meaner, sharper — they ate the 'Candidate' confusion alive.  

Their secret?  
They hyperfixated on the **top 3-4 features**, especially the flags.  
Efficiency level: **“work smarter, not harder.”**

---

## `> THE STRESS TEST // Enter the TESS data`

```
>> NEW DATASET DETECTED: TOI_dataset.csv
>> RUNNING SIMULATION...
```

We got cocky.  
So we dropped in **TESS** — the next-gen NASA mission data — to see if our Kepler-trained models could survive new cosmic turf.

Result?  
Our **OG Random Forest** choked.  
Accuracy crashed to `~62%`.

Why?  
TESS didn’t have the same `false_positive_flags` our model was addicted to.  
Without its favorite cheat codes, the model panicked and called everything a *Candidate*.  
Turns out, it wasn’t a planet hunter — just a good test-taker.  
💀 *Lesson learned.*

---

## `> aight, run it`

Wanna boot this system yourself?  
Here’s the blueprint.

### // The Kepler Saga
```
analysis_v2.py                -> Cleans the data. Makes it shine. ✨
feature_importance_v1.py      -> Calls out the fakes. Savage.
confirmed_vs_candidate_v1.py  -> Finds the chosen ones.
multiclass_model_v1.py        -> The all-in-one OG model.
multiclass_gradient_boosting_v1.py -> The remix. The better one. 👑
multiclass_xgboost_v1.py      -> The rival champion. Also 👑.
```

---

### // The TESS Expansion
```
translate_toi_data_v2.py      -> Translates TESS into Kepler-speak.
test_rf_on_toi.py             -> Stress-tests the OG model... and watches it implode. 🔥
```

All charts and visual receipts live in the `/charts` folder.  
Open them if you dare. Or don’t. The void doesn’t care.

---

## `> dev corner // OPERATOR LOGS`

```
>> ACCESS LEVEL: INTERNAL
>> AUTHENTICATION COMPLETE
```

### 🧠 Model Specs
- **Random Forest:** 200 estimators, depth-tuned for precision.  
- **Gradient Boosting:** Learning rate optimized via grid search.  
- **XGBoost:** Used GPU acceleration and early stopping to avoid overfitting.  
- **Scaler:** `StandardScaler()` for signal normalization across Kepler metrics.  

### 🔍 Key Features
- `koi_fpflag_nt`: False positive not transit-like flag.  
- `koi_fpflag_ss`: False positive due to stellar system.  
- `koi_depth`, `koi_duration`, `koi_period`: Core orbital metrics.  
- `koi_prad`: Planetary radius estimate.  
- `koi_model_snr`: The real MVP — signal-to-noise ratio that separates legends from noise.

### 🧩 Tools & Libraries
```
Python 3.11
scikit-learn 1.5
xgboost 2.1
pandas, numpy, matplotlib, seaborn
tqdm (for that hacker-progress-bar aesthetic)
```

---

## `> data junkie notes // transmission logs`

```
>> SYSTEM NOTE: NO DATA LEFT BEHIND
```

- **Kepler Data:** Pulled from NASA Exoplanet Archive.
  Cleaned, imputed missing values, filtered high-quality transit signals only.
- **TESS TOI Data:** Re-aligned to Kepler feature schema using `translate_toi_data_v2.py`.
  Missing columns were interpolated or zero-padded for testing cross-mission generalization.
- **Feature Importance:**
  Gradient Boosting ranked the `false_positive_flags` as the top indicators — proving bias toward flag-based metadata rather than pure signal metrics.

---

## `> status: SYSTEMS NOMINAL`

```
[CONNECTION STABLE]
[ALL MODELS DEPLOYED]
[WAITING FOR NEXT TRANSMISSION...]
```

### `══[ End of Line // KEPLER OUT ]══`
### `> Transmission Complete`