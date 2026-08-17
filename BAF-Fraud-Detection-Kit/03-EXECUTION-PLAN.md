# The Execution Plan — hour by hour

A hackathon is a time-allocation problem, not a modelling problem. Most teams
lose by spending 80% of their time chasing the fourth decimal place of AUC and
20% on everything that actually earns marks.

**Assume the score gets you shortlisted. The story wins the room.**

---

## The 60/40 rule

| | |
|---|---|
| **~60% of your time** | A correct, defensible model. Correct means *temporally validated* and *honestly measured*. Not maximal. |
| **~40% of your time** | Fairness analysis, explainability, the writeup, and the demo. This is where you separate from the field. |

Almost every team inverts this. That is your opening.

---

## Hour 0 → 1 — Reconnaissance. Do not train anything.

The single most expensive mistake available to you is building your validation
wrong, because every number you produce afterwards is then meaningless.

1. Download the data. While it downloads, run the toolkit on synthetic data so
   your pipeline is already debugged:
   ```bash
   pip install lightgbm pandas scikit-learn matplotlib shap
   python make_sample.py --rows 60000 --out sample_train.csv
   python run_pipeline.py --train sample_train.csv
   ```
2. Run the §4 checks in `01-DATASET-BIBLE.md` on the **real** files.
3. **Confirm the split** — 10 seconds, and it determines everything downstream:

   ```python
   print(sorted(train.month.unique()), sorted(test.month.unique()))
   ```

   > **Already verified for you:** the organisers state they *"randomly chosen
   > 700,000 accounts (70%) as the training data and made the remaining
   > 300,000 accounts (30%) test data."* It is a **RANDOM** split and `month`
   > is in both files.
   >
   > So: **stratified K-fold, keep `month`, train on everything.** This is the
   > opposite of what every generic BAF tutorial says, because those follow the
   > NeurIPS paper's temporal protocol. Run the check anyway and say in your
   > writeup that you verified it — that is a point of difference in itself.

   Getting this wrong is how a 0.90 local score becomes a 0.78 leaderboard score.

**Deliverable:** you know your split, your metric (**ROC-AUC**), and your dead columns.

### Know the bar before you start

| | AUC |
|---|---|
| Winning score | **0.90444** |
| 3rd place (took 3 submissions) | 0.90108 |
| 7th | 0.83518 |

**Target 0.905. Floor 0.89. Above 0.92 means you have leakage — go find it.**

Only 8 teams entered, 47 submissions total, and the private board preserved the public ordering exactly. This is winnable with a clean model plus the differentiators below.

---

## Hour 1 → 3 — Baseline, submitted early

Submit something to the leaderboard within the first three hours. An early
submission de-risks the format, confirms the metric, and removes the 3am panic
where nothing serialises.

```bash
python run_pipeline.py --train train.csv --test test.csv --submit submission.csv --id-col <id>
```

- Preprocessing correct (sentinels → NaN + indicators, constants dropped)
- LightGBM with sane parameters
- Temporal validation
- ROC-AUC, PR-AUC and TPR@5%FPR all reported

**Deliverable:** a leaderboard score and a validation number you trust. Write both down. If they disagree badly, your split is wrong — fix that before anything else.

---

## Hour 3 → 6 — The ablation. This is your evidence, not your tuning.

Your objective statement says "data balancing". Do not obey it blindly and do
not ignore it. **Test it and report the result.**

`run_pipeline.py` already runs: no balancing vs `scale_pos_weight` vs
undersampling with prior correction. Add SMOTE if you have time
(`pip install imbalanced-learn`) so nobody can say you didn't try it.

Record the table. It goes straight onto a slide.

> **Judges reward the team that tested the assumption over the team that
> followed it.** If balancing doesn't help your ranking metric, saying so —
> with a table — is a stronger result than a fabricated improvement. See
> `research/R2-imbalance-truth.md` for the argument and the evidence.

**Deliverable:** an ablation table, and a defensible decision with a reason.

---

## Hour 6 → 10 — Features and tuning

Now, and only now, chase score.

1. **Feature engineering** — the catalogue is in `research/R3-features-and-tuning.md`, and `baf.add_features()` already implements the highest-value set: velocity burst ratios, the email/name coherence cluster, thin-file scores, contactability, credit-limit coherence.
2. **Ablate your features too.** Run with and without `add_features()`. If they add nothing, drop them and say so.
3. **Tune** — a modest random/Optuna search over `num_leaves`, `min_child_samples`, `feature_fraction`, `learning_rate`. On a 1% positive rate, `min_child_samples` matters most: small leaves memorise noise.
4. **Blend** — averaging LightGBM + XGBoost + CatBoost ranks usually buys a small, reliable gain. Average *ranks*, not probabilities, when the metric is AUC.

**Stop tuning when gains fall below ~0.002 AUC.** That is noise, and the time
is worth more elsewhere.

**Deliverable:** your final model, and honest before/after numbers for each change.

---

## Hour 10 → 14 — The differentiators

This is the part almost nobody does, and it is worth more than the tuning.

1. **Fairness.** Run `baf.fairness_report()`. BAF was *purpose-built* for this: `customer_age` is the protected attribute, predictive equality is the metric, and the paper's own finding is that strong models falsely flag older applicants substantially more often. Reproduce that on your model. Then show a mitigation and the trade-off it costs you. Details in `research/R4-fairness-explainability-pitch.md`.
2. **Explainability.** Global SHAP for the model, and — more impressive — a **per-application reason code**: for one declined applicant, the three factors that drove the decision. That is what a bank must legally provide.
3. **The operating point.** Don't hand over a probability. Say: *"At a 5% false-positive budget, this model catches X% of fraud. That is N reviews per day for a team that can handle M."* Business framing beats a metric.
4. **Temporal robustness.** Score month 6 and month 7 separately. If performance holds, that is a reliability claim nobody else will make.

**Deliverable:** three or four plots that each make a point.

---

## Hour 14 → 18 — Build the story

Write the narrative before you make slides.

**The thesis:**

> *Fraud detection is not an accuracy problem. On this data, predicting "never
> fraud" scores 98.9% accuracy and catches zero criminals. The real problem is
> that every mistake has a victim — a missed fraud costs the bank, and a false
> positive denies a real person a bank account. We built a model that is
> honest about both, and we measured who pays for its errors.*

Slide outline:

1. **The 98.9% trap** — open with it. Immediately establishes you understand imbalance.
2. **The data and its traps** — sentinel missingness, temporal drift. Shows depth.
3. **Validation done right** — the temporal split, and why random CV lies here.
4. **The balancing ablation** — your table. You tested the brief instead of assuming it.
5. **Results** — ROC-AUC, PR-AUC, TPR@5%FPR. Never accuracy alone.
6. **Who pays for the errors** — the fairness finding. This is your moment.
7. **Explaining a decline** — one real reason code.
8. **What we'd do with more time** — shows judgement.

---

## Hour 18 → 20 — Rehearse and harden

- **Re-run the whole pipeline from a clean state.** If it doesn't reproduce, it doesn't exist.
- Pin versions in `requirements.txt`.
- Rehearse out loud, timed. Cut 20%.
- Prepare for the five hardest questions (`research/R4-...`).
- Have a screenshot of every plot in case the laptop misbehaves.

---

## The five sentences that win the room

Memorise these. Each signals expertise in one line.

1. *"Accuracy is meaningless here — the do-nothing baseline is 98.9%."*
2. *"We split temporally, months 0–5 to 6–7, because random CV leaks the future and this dataset has real prevalence drift."*
3. *"The brief said to use data balancing, so we tested it — here's the ablation. It didn't improve our ranking metric, and here's why that's expected."*
4. *"We report TPR at 5% FPR because that's the operating point a bank actually buys — every false positive is a real customer wrongly rejected."*
5. *"We also measured who pays for those errors. Our model falsely flags older applicants more often, and here's the trade-off curve for fixing it."*

---

## Things that lose

- Reporting accuracy on a 1.1% positive class.
- A 0.99 AUC you cannot explain. It is leakage. A judge will ask, and "I'm not sure" ends your run.
- SMOTE applied *before* the train/validation split — this leaks synthetic points derived from validation rows into training. If you use SMOTE, it goes **inside** the training fold only.
- A pipeline that only runs in one notebook kernel with hidden state.
- Spending the last two hours tuning instead of rehearsing.
