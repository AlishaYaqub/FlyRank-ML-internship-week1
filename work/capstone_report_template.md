# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Alisha Yaqub
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/AlishaYaqub/FlyRank-ML-internship-week1
- **Date:** September 2026

## 0. Abstract

Which pages should an editor review first when search performance is declining, and can a simple model beat a hand-written rule at ranking them? I built a client-grouped, honest baseline rule on FlyRank's search warehouse data, then trained Logistic Regression and Random Forest models on the same features and the same split. Random Forest reached 90% precision@100 versus the baseline rule's 40% and a random base rate of 19%, more than doubling the baseline's usefulness. The output is a ranked action queue with reason codes an editor can use directly tomorrow. The model is most reliable on higher-traffic pages and least reliable on very low-traffic pages, where the decline label itself becomes noisy.

## 1. Problem framing

The unit of analysis is one page (`content_id`), evaluated over a mid-panel month (March 2026). The decision this supports: out of thousands of pages, which should a content editor review first for a possible refresh? The output is a ranked queue, where each page carries a score, a reason code, and an action label (`review_now` / `monitor` / `no_action`). The cost of a wrong call runs in both directions: a false positive wastes an editor's limited review time, while a false negative leaves a genuinely declining page unreviewed. ML earns its place here because no single signal cleanly separates declining pages from healthy ones — position tier alone showed decline rates ranging non-monotonically from 24% to 61% across tiers, meaning a single-column threshold rule would misclassify a large share of pages either way.

## 2. Data safety

I used `fact_content_daily_performance` (the `month=2026-03` partition) joined with `dim_content` for content metadata (`content_updated_date`, `content_created_date`). I deliberately excluded `trend_direction` and `trend_pct` as features, since these are label-derived and would leak the answer directly into the inputs — they were used only to help understand the label in the framing stage, never as a model feature. I also excluded `sessions_paid`, `sessions_referral`, and `sessions_social`, since this lane is specifically about organic search decline, not other traffic channels. `content_hash_id` and `client_hash_id` are used only for joining and grouping (e.g. the client-grouped train/test split), never as model inputs, since they are pseudonymous identifiers with no predictive meaning of their own. No client names, domains, URLs, or raw exports appear anywhere in this repo's `work/` folder.

## 3. Baseline

My baseline is a transparent, hand-written rule built in Week 4. Before writing it, I tested two candidate signals directly against the data rather than assuming they worked:

- **Staleness** (days since a page was last updated, linked to FlyRank's real refresh flag): **FALSE**. Decline rate was nearly flat across all staleness buckets (0.155–0.201), with no clear upward trend as pages aged.
- **CTR vs. position-tier mismatch** (linked to FlyRank's real CTR-fix logic): **OPPOSITE** of the assumption. Pages underperforming their tier's average CTR actually declined *less* often (11.2%) than pages at or above average CTR (29.9%) — likely because currently-declining pages have low volume, which makes their CTR noisy and sometimes coincidentally high.

Since staleness showed no relationship, it was dropped. The baseline rule instead prioritizes pages with high impressions and CTR at or above their position tier's average — the group Signal 2 showed actually declines more. On the same client-grouped test split used for the model (Section 5), this rule reaches **40% precision@100**, more than double the 19% random base rate, making it a fair, non-trivial comparison point for the model.

## 4. Model / analysis

Since the target is a yes/no outcome with an observed label, I started with Logistic Regression, then compared it against Random Forest to check whether added complexity actually earned its place. The target is a proxy: `is_declining = late_clicks < early_clicks`, comparing each page's clicks in the first half of March against the second half — an observed measured outcome, not a hand-defined rule. Features used: `early_clicks`, `early_impressions`, `early_avg_position`, `early_engaged_sessions`, `early_scroll_events`, and `ctr` — all computed strictly from data available before the March 15 cutoff. `late_clicks` (the column the label is built from) was deliberately never used as a feature.

## 5. Evaluation

I split by **client**, not by page, since pages from the same client can share management patterns (e.g. one team managing many pages the same way), which would let information leak across train and test under a random page-level split. The split produced 35 training clients and 12 test clients with zero client overlap (133,474 training rows, 43,264 test rows).

| Method | Precision@100 |
|---|---|
| Random Forest | 0.900 |
| Logistic Regression | 0.690 |
| Baseline rule (Week 4) | 0.400 |
| Base rate (random) | 0.192 |

Both models clearly beat the baseline rule and the random base rate, with Random Forest the strongest. Looking at individual wrong predictions, all involved pages with very low `early_clicks` (1–10) — at that volume, a single click's difference can flip the `is_declining` label entirely, making these genuinely hard, noisy cases rather than clear model failures.

## 6. Interpretation

Feature importance for Random Forest: `ctr` (0.486), `early_clicks` (0.364), `early_impressions` (0.103), `early_scroll_events` (0.021), `early_avg_position` (0.020), `early_engaged_sessions` (0.007). No single feature dominates completely, which is a reasonable sign against obvious leakage. However, `ctr` and `early_clicks` together account for roughly 85% of the model's decisions, which deserves an honest caveat: since `is_declining` was defined by comparing `late_clicks` to `early_clicks`, pages with high `early_clicks` mechanically have more room to "decline" in absolute terms. Part of the model's strength may reflect this built-in relationship rather than a purely new business signal. This is a limitation of the label definition, not evidence of literal leakage (no future-window or label-derived column was used as an input), but it is worth stating plainly rather than presenting the 90% figure without context.

The clearest negative result of this project is Signal 1 (staleness): a genuinely well-understood "no effect," which saved the baseline rule from being built on a non-working assumption, and is itself a useful finding for FlyRank's refresh-flag logic.

## 7. Recommendation

The ranked queue (`work/outputs/baseline_action_score.csv`, regenerated by the notebook on each run) gives an editor a directly usable priority list: `review_now` for the top-scoring pages, `monitor` for mid-scoring pages, `no_action` for the rest. Each row carries a reason code (`HIGH_VALUE_AT_RISK` or `LOW_VOLUME_SKIP`) explaining why it was flagged. My confidence is highest for higher-traffic pages, where the label and the model's signal are both more stable; confidence is lowest for very low-traffic pages, where a single click can flip the outcome. I would recommend FlyRank treat the model's top-100 list as a strong starting point for weekly review, while treating low-volume flagged pages as lower-confidence and worth a lighter-touch check rather than a full rewrite commitment.

## 8. Reproducibility

To re-run this work from a fresh clone: open `work/notebooks/w03_data_contract.ipynb` through `w05_model.ipynb` in Google Colab, add a `HF_TOKEN` secret (a Hugging Face Read token with access to `FlyRank/internship-warehouse`), and run each notebook top to bottom (Runtime → Run all). All random processes use `random_state=42` for the train/test split and both models, so results are reproducible on a fresh run against the same `month=2026-03` partition. Package versions used: `duckdb`, `pandas`, `scikit-learn` (installed fresh via `%pip install` at the top of each notebook, no pinned versions beyond what Colab provides by default at run time).

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

> **Claims checklist:** all language above uses observed / measured / directional / decision-support framing. No causal claims are made about *why* pages decline, only observed associations. No claim is made about predicting or reverse-engineering Google's ranking algorithm. No client names, domains, URLs, or credentials appear anywhere in this report or the underlying notebooks. Base rate (19.2%) is reported alongside precision@100 so the headline numbers are not misread as inflated by class imbalance alone.
