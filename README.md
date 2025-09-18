### `> K E P L E R // a vibe check`

Yo. So you stumbled on this project. We're diving into NASA's data to sort the legit exoplanets from the space trash. It's a whole mood. We're here to expose the fakes and crown the real ones. Straight up.

---

### `> the game plan`

So like, how'd we do it? It's a whole saga.

1. **DROP THE FAKES.** 🚮
    First, we built a model to spot the posers—the "False Positives." It learned their tells, their bad vibes, all of it. If it ain't a planet, it's gotta go.

2. **CROWN THE REAL ONES.** 👑
    Then, for the ones left, we got picky. We trained another model to figure out what separates a "Confirmed" planet from a "Candidate" that just doesn't make the cut. This is the real glow up.

3. **THE FINAL BOSS.** 👾
    After all that, we were like, "what if we just mashed it all together?" So we built one big multi-class model to do everything at once. Then we built a _second_, stronger one just to see if we could. It's called main character energy.

---

### `> spill the tea ☕`

So what did we find out? Here's the breakdown.

#### `// how to spot a poser`

Turns out the fakes are loud about it. They give themselves away if you know the signs.

- **They're Flagged:** The data literally has `false_positive_flags`. Our model just had to pay attention. 💅
- **They're HUGE:** They show up way too big, probably just some other star trying to steal the spotlight.
- **They're Shady:** Their light signal is all over the place, not even centered on the star. Looks pretty suspicious.

#### `// the glow up: 'maybe' to 'definitely'`

The difference between a _maybe_ and a _hell yes_? It's all about the quality.

- **The Signal is Solid:** A real one has a transit signal that's strong, clear, and unmistakable. No static, all facts.
- **Zero Doubts:** The real tea is in the details. A confirmed planet has measurements with tiny, almost non-existent error bars. It's not confident, it's **certain**. We don't do "idk" around here.

---

### `> THE REMIX // RF vs. GB`

So about that final boss model. Our first attempt (Random Forest) was solid. It hit **92.6% accuracy**. Not bad.

But then we brought in the challenger: **Gradient Boosting**. And it delivered.

The new model cranked the accuracy up to **94.2%**. It was way better at spotting the 'Candidates'—the ones the first model kept tripping over. The new king was also pretty straightforward. It got obsessed with the top 3-4 features, especially the flags, and just relied on them for the win. Work smarter, not harder.

---

### `> aight, run it.`

Wanna see the magic? Fine.

The whole story is in the scripts. Just run 'em in order. Don't mess it up.

1. `analysis_v2.py` -> Cleans the data, makes it look pretty.
2. `feature_importance_v1.py` -> The part where we call out the fakes.
3. `confirmed_vs_candidate_v1.py` -> The part where we find the real ones.
4. `multiclass_model_v1.py` -> The all-in-one model. The OG.
5. `multiclass_gradient_boosting_v1.py` -> The remix. The better one. 👑

All the visuals get dumped in the `/charts` folder. Go look. Or don't. Whatever.
