---

### `> K E P L E R // a vibe check`

`________________________________________________`

Yo. So you stumbled on this project. We're diving into NASA's data to sort the legit exoplanets from the space trash. It's a whole mood. We're here to expose the fakes and crown the real ones. No cap.

---

### `> the game plan`

`________________`

So like, how'd we do it? Easy.

1.  **YEET THE FAKES.** 🚮
    First, we built a model to spot the posers—the "False Positives." It learned their tells, their bad vibes, all of it. If it ain't a planet, it's gotta go. Simple as.

2.  **CROWN THE QUEENS.** 👑
    Then, for the ones left, we got picky. We trained another model to figure out what separates a "Confirmed" baddie from a "Candidate" that's just... mid. This is the real glow up.

---

### `> spill the tea ☕`

`______________________`

So what did we find out? IYKYK.

#### `// how to spot a poser`

Turns out the fakes are loud about it. They give themselves away if you know the signs.

- **They're Flagged:** The data literally has `false_positive_flags`. Our model just had to not be blind. 💅
- **They're HUGE:** They got that main character syndrome, showing up way too big. Probs just some other star trying to steal the spotlight.
- **They're Shady:** Their light signal is all over the place, not even centered on the star. It's giving... suspicious.

#### `// the glow up: 'maybe' to 'yas queen'`

The difference between a _maybe_ and a _hell yes_? It's all about the quality.

- **That Signal Slaps:** A real one has a transit signal that's strong, clear, and unmistakable. No static, all facts.
- **Zero Doubts:** The real tea is in the deets. A confirmed planet has measurements with tiny, almost non-existent error bars. It's not confident, it's **certain**. We don't do "idk" around here.

---

### `> aight, bet. run it.`

`__________________________`

Wanna see the magic? Fine.

The whole story is in the scripts. Just run 'em in order. Don't mess it up.

1.  `analysis_v2.py` -> Cleans the data, makes it look pretty.
2.  `feature_importance_v1.py` -> The part where we call out the fakes.
3.  `confirmed_vs_candidate_v1.py` -> The part where we find the main characters.

All the sick visuals get dumped in the `/charts` folder. Go look. Or don't. Whatever.
