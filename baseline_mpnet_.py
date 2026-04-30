import warnings
import sys
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from pipeline import (
    log, close_log,
    build_sbert_model,
    domain_adapt_mlm,
    run_row_by_row,
    run_cross_validation,
    apply_pseudo_labels,
    retrain_with_pseudo,
)

# base_path = "/content/drive/MyDrive/Colab_Notebooks/a5/"
base_path = "./"
CFG = {
    "input":        base_path + "train_2022.csv",
    "test":         base_path + "test_no_answer_2022.csv",
    "test_output":  base_path + "result_baseline.csv",

    "model":        "rf",

    "skip_da":      False,
    "load_da":      None,
    "da_epochs":    1,
    "da_batch":     8,
    "da_lr":        3e-5,
    "da_mlm_prob":  0.15,
    "da_save":      base_path + "baseline_da_model.pt",

    "skip_ft":      False,

    "ft_epochs":    3,
    "ft_lr":        2e-5,
    "ft_batch":     16,
    "ft_pairs":     3000,
    "ft_warmup":    100,
    "ft_save":      base_path + "baseline_ft_model.pt",

    # Pseudo-label settings
    # pseudo_iters=1  → A+B (one-shot: retrain SBERT + classifier on high-conf test)
    # pseudo_iters>1  → C   (iterative: repeat A+B multiple rounds)
    "pseudo_iters":       1,
    "pseudo_thresh_low":  0.15,   # score <= this → label 0
    "pseudo_thresh_high": 0.85,   # score >= this → label 1
}


def main():
    c = CFG

    from pipeline import SBERT_MODEL_NAME, SBERT_DIM
    sep = "=" * 60
    log(f"\n{sep}")
    log("  Sentiment Scoring Pipeline  [v8 / no-arg]")
    log("  Two-Phase Fine-Tuning: MLM + CV-internal CosineSimilarityLoss")
    log(sep)
    log(f"  Input      : {c['input']}")
    log(f"  Test       : {c['test']}")
    log(f"  Model      : {c['model']}")
    log(f"  Encoder    : {SBERT_MODEL_NAME}  ({SBERT_DIM}-dim)")
    if c["load_da"]:
        phase1_desc = f"LOAD from {c['load_da']}"
    elif c["skip_da"]:
        phase1_desc = "SKIPPED"
    else:
        phase1_desc = f"MLM  epochs={c['da_epochs']}  lr={c['da_lr']}  mlm_prob={c['da_mlm_prob']}"
    log(f"  Phase 1    : {phase1_desc}")
    log(f"  Phase 2    : CosineSimilarityLoss (inside CV)  epochs={c['ft_epochs']}  lr={c['ft_lr']}  pairs={c['ft_pairs']}")
    log(f"  Pseudo-L   : iters={c['pseudo_iters']}  thresh=[≤{c['pseudo_thresh_low']}, ≥{c['pseudo_thresh_high']}]\n")

    train_df = pd.read_csv(c["input"])
    train_df["row_id"] = train_df["row_id"].astype(int)
    train_df["LABEL"]  = train_df["LABEL"].astype(int)
    log(f"  Loaded {len(train_df)} training rows")

    test_df = pd.read_csv(c["test"])
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    log(f"  Loaded {len(test_df)} test rows\n")

    log("── Initialising models ──")
    sbert = build_sbert_model()

    if c["load_da"]:
        log(f"\n  [Phase 1] 載入已存 checkpoint: {c['load_da']}\n")
        sbert = SentenceTransformer(c["load_da"])
    elif not c["skip_da"]:
        all_texts = (
            train_df["TEXT"].astype(str).tolist() +
            test_df["TEXT"].astype(str).tolist()
        )
        domain_adapt_mlm(
            sbert, all_texts,
            epochs=c["da_epochs"], batch_size=c["da_batch"],
            lr=c["da_lr"], mlm_prob=c["da_mlm_prob"],
            output_path=c["da_save"],
        )
    else:
        log("\n  [Phase 1 skipped — using base model weights]\n")

    if c["skip_ft"]:
        log("  [skip_ft=True，Phase 1 完成後退出]")
        close_log()
        return

    log("\n── Step 3: 5x 80/20 CV with fine-tune inside each round ──")
    row_scores, trained_pipe, sbert_final, feat_cols = run_cross_validation(
        train_df, sbert, c["model"], cfg=c, n_splits=5
    )

    log(f"\n── Processing test set ──")
    test_agg_df = run_row_by_row(test_df, sbert_final, has_label=False)
    X_test      = test_agg_df[feat_cols].values
    test_scores = np.clip(trained_pipe.predict(X_test), 0, 1)

    # ── Pseudo-label phase (A+B when iters=1, iterative C when iters>1) ──────
    n_iters = c.get("pseudo_iters", 0)
    if n_iters > 0:
        log(f"\n── Pseudo-Label Phase  (iters={n_iters}, "
            f"thresh=[≤{c['pseudo_thresh_low']}, ≥{c['pseudo_thresh_high']}]) ──")
        for it in range(1, n_iters + 1):
            log(f"\n  ══ Pseudo-Label Iteration {it}/{n_iters} ══")
            augmented_df, n_added = apply_pseudo_labels(
                train_df, test_df, test_scores,
                c["pseudo_thresh_low"], c["pseudo_thresh_high"],
            )
            if n_added == 0:
                log("  No samples passed threshold — stopping early.")
                break

            log("  Retraining SBERT (B) + classifier (A) on augmented data …")
            new_pipe, new_sbert = retrain_with_pseudo(augmented_df, sbert, c["model"], cfg=c)

            test_agg_df = run_row_by_row(test_df, new_sbert, has_label=False)
            X_test      = test_agg_df[feat_cols].values
            new_scores  = np.clip(new_pipe.predict(X_test), 0, 1)
            log(f"  Score shift — mean: {test_scores.mean():.4f} → {new_scores.mean():.4f}  "
                f"std: {test_scores.std():.4f} → {new_scores.std():.4f}")

            test_scores  = new_scores
            trained_pipe = new_pipe
            sbert_final  = new_sbert

        log("\n  Pseudo-label phase complete.\n")

    test_labels = (test_scores >= 0.5).astype(int)
    test_output = test_agg_df[["row_id"]].copy()
    test_output["LABEL"] = test_labels
    test_output.to_csv(c["test_output"], index=False)
    log(f"  Saved test predictions to: {c['test_output']}")

    close_log()


if __name__ == "__main__":
    import argparse, ast
    parser = argparse.ArgumentParser(description="Sentiment Scoring Pipeline")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="Override CFG keys, e.g. --set model=gbm ft_epochs=5 skip_da=True")
    args = parser.parse_args()
    if args.set:
        for kv in args.set:
            key, val = kv.split("=", 1)
            try:
                CFG[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                CFG[key] = val
    main()
