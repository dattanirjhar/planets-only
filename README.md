# `> K E P L E R // a vibe check`

Yo. So you stumbled on this project. We're diving into NASA's data to sort the legit exoplanets from the space trash. It's a whole mood. We're here to expose the fakes and crown the real ones. Straight up.

## `> the game plan`

So like, how'd we do it? It's a whole saga.

### DROP THE FAKES. 🚮

    First, we built a model to spot the posers—the "False Positives." It learned their tells, their bad vibes, all of it. If it ain't a planet, it's gotta go.

### CROWN THE REAL ONES. 👑

    Then, for the ones left, we got picky. We trained another model to figure out what separates a "Confirmed" planet from a "Candidate" that just doesn't make the cut. This is the real glow up.

### THE FINAL BOSS. 👾

    After all that, we were like, "what if we just mashed it all together?" So we built one big multi-class model to do everything at once. Then we brought in some challengers to see if they could do better. It's called main character energy.

## `> spill the tea ☕`

So what did we find out? Here's the breakdown.

### // how to spot a poser

Turns out the fakes are loud about it. They give themselves away if you know the signs.

    They're Flagged: The data literally has false_positive_flags. Our model just had to pay attention. 💅

    They're HUGE: They show up way too big, probably just some other star trying to steal the spotlight.

    They're Shady: Their light signal is all over the place, not even centered on the star. Looks pretty suspicious.

### // the glow up: 'maybe' to 'definitely'

The difference between a maybe and a hell yes? It's all about the quality.

    The Signal is Solid: A real one has a transit signal that's strong, clear, and unmistakable. No static, all facts.

    Zero Doubts: The real tea is in the details. A confirmed planet has measurements with tiny, almost non-existent error bars. It's not confident, it's certain. We don't do "idk" around here.

## `> THE REMIX // RF vs. The Champs`

So about that final boss model. Our first attempt (Random Forest) was solid. It hit 92.6% accuracy. Not bad.

But then we brought in the challengers: Gradient Boosting and XGBoost. And they both delivered, big time.

Both new models cranked the accuracy up to a clean 94.2%. They were way better at spotting the 'Candidates'—the ones the first model kept tripping over. Their strategy was also pretty straightforward: get obsessed with the top 3-4 features, especially the flags, and just rely on them for the win. Work smarter, not harder.

## `> THE STRESS TEST // Enter the TESS data`

Just when we thought we had the perfect models, we decided to push them. We grabbed a whole new dataset from the TESS mission (TOI_dataset.csv) to see if our Kepler-trained models could hang in a new environment.

And our OG Random Forest model? It got exposed.

The accuracy tanked to ~62%. It turns out the TESS data is missing the false_positive_flags our model loved so much. Without its favorite cheat codes, the model pretty much gave up and started guessing everything was a 'Candidate'. It proved our first model wasn't a true planet hunter; it just learned to pass one specific test. Oof.

## `> aight, run it`

Wanna see the magic? Fine. The whole story is in the scripts. Don't mess it up.

### // The Kepler Saga

    analysis_v2.py -> Cleans the data, makes it look pretty.

    feature_importance_v1.py -> The part where we call out the fakes.

    confirmed_vs_candidate_v1.py -> The part where we find the real ones.

    multiclass_model_v1.py -> The all-in-one model. The OG.

    multiclass_gradient_boosting_v1.py -> The remix. The better one. 👑

    multiclass_xgboost_v1.py -> The other better one. Also 👑.

### // The TESS Expansion

    translate_toi_data_v2.py -> Translates the raw TESS data so our models can read it.

    test_rf_on_toi.py -> The script that stress-tests the OG model and watches it fail.

All the visuals get dumped in the /charts folder. Go look. Or don't. Whatever.
