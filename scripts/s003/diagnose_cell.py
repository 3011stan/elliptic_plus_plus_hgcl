#!/usr/bin/env python3
"""Diagnostic script for inspecting and isolating matrix cell execution in Study 003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from hgcl.studies.s003.artifacts import EmbeddingCacheKey
from hgcl.studies.s003.baselines import AdapterContext, GraphData
from hgcl.studies.s003.config import load_config
from hgcl.studies.s003.matrix import (
    EmbeddingCacheStore,
    MatrixScheduler,
    create_adapter_for_cell,
    embedding_cache_key_for_cell,
    execute_matrix_cell,
    parse_cell_key,
)
from hgcl.studies.s003.pipeline import find_compatible_approval, load_prepared
from hgcl.studies.s003.splits import build_label_budgets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose a specific Study 003 matrix cell in isolation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "s003" / "lab.yaml",
        help="Path to lab config (default: configs/s003/lab.yaml)",
    )
    parser.add_argument(
        "--cell",
        type=str,
        default="representation:h_only:0.01:71",
        help="Canonical cell key to test (default: representation:h_only:0.01:71)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "s003",
        help="Artifacts root directory (default: artifacts/s003)",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=None,
        help="Prepared dataset directory (default: auto-detected under artifacts/s003/prepared)",
    )
    parser.add_argument(
        "--ssl-epochs",
        type=int,
        default=None,
        help="Override SSL epochs (default: from config)",
    )
    parser.add_argument(
        "--downstream-epochs",
        type=int,
        default=None,
        help="Override downstream epochs (default: from config)",
    )
    parser.add_argument(
        "--check-cache-only",
        action="store_true",
        help="Only inspect dataset, cache, and label budgets without running training",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Abort if cached embeddings are not found (prevents heavy SSL training)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("S003 Cell Diagnostic Tool")
    print("=" * 80)
    print(f"Target cell key : {args.cell}")
    print(f"Config path     : {args.config}")
    print(f"Artifacts root  : {args.artifacts_root}")

    # 1. Load config
    if not args.config.is_file():
        print(f"ERROR: Config file not found at {args.config}")
        return 1
    config = load_config(args.config)
    ssl_epochs = args.ssl_epochs or int(config.raw["ssl"]["epochs"])
    downstream_epochs = args.downstream_epochs or int(config.raw["downstream"]["epochs"])
    print(f"SSL epochs      : {ssl_epochs}")
    print(f"Downstream ep.  : {downstream_epochs}")

    # 2. Find prepared dataset directory
    prep_dir = args.prepared_dir
    if prep_dir is None:
        prepared_parent = args.artifacts_root / "prepared"
        if not prepared_parent.is_dir():
            print(f"ERROR: Prepared parent dir not found at {prepared_parent}")
            return 1
        candidates = [p for p in prepared_parent.iterdir() if p.is_dir()]
        if not candidates:
            print(f"ERROR: No prepared dataset directories found in {prepared_parent}")
            return 1
        prep_dir = sorted(candidates)[0]
    print(f"Prepared dir    : {prep_dir}")

    # 3. Load prepared dataset
    print("\nLoading prepared dataset...")
    prepared = load_prepared(prep_dir, config)
    print(f"Loaded snapshots : {len(prepared.snapshots)} (training: {len(prepared.training_snapshots)})")
    print(f"Data digest      : {prepared.data_digest}")

    # 4. Check git revision and approval
    import subprocess
    try:
        git_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        revision_str = git_res.stdout.strip()
    except Exception:
        revision_str = "0" * 40
    print(f"Git revision     : {revision_str}")

    try:
        approval = find_compatible_approval(
            args.artifacts_root,
            data_digest=prepared.data_digest,
            lab_config_digest=config.digest,
            code_revision=revision_str,
            design_digest="0" * 64,  # relaxed for diagnostic
            require_reverse_edge_ablation=False,
        )
        effective_revision = approval.code_revision if approval else revision_str
    except Exception:
        effective_revision = revision_str
    print(f"Effective rev    : {effective_revision}")

    # 5. Check cache directory and existing embeddings
    cache_root = args.artifacts_root / "cache" / "embeddings"
    print(f"\nEmbedding cache  : {cache_root}")
    cache_files = list(cache_root.glob("*.pt")) if cache_root.is_dir() else []
    print(f"Cached files cnt : {len(cache_files)}")
    for cf in cache_files:
        print(f"  - {cf.name} ({cf.stat().st_size / 1024 / 1024:.2f} MB)")

    cache_store = EmbeddingCacheStore(root=cache_root)

    # 6. Resolve target cell key and expected cache key
    parsed = parse_cell_key(args.cell)
    print(f"\nCell breakdown   : family={parsed.family}, method={parsed.method}, variant={parsed.variant}, fraction={parsed.fraction}, seed={parsed.seed}")

    expected_cache_key = embedding_cache_key_for_cell(
        args.cell,
        data_digest=prepared.data_digest,
        pretraining_config_digest=config.digest,
        code_revision=effective_revision,
    )
    print(f"Expected cache digest: {expected_cache_key.digest}")
    cached_payload = cache_store.get(expected_cache_key)
    print(f"Cache hit on digest  : {cached_payload is not None}")

    if cached_payload is not None:
        if isinstance(cached_payload, dict):
            embs = cached_payload.get("embeddings")
            mod = cached_payload.get("model")
            print(f"Cached embeddings shape: {getattr(embs, 'shape', None)}, dtype: {getattr(embs, 'dtype', None)}")
            if embs is not None and torch.is_tensor(embs):
                print(f"  NaNs present : {bool(torch.isnan(embs).any())}")
                print(f"  Infs present : {bool(torch.isinf(embs).any())}")
                print(f"  Min / Max    : {float(embs.min()):.4f} / {float(embs.max()):.4f}")
            print(f"Cached model type      : {type(mod)}")
        else:
            print(f"Cached raw payload type: {type(cached_payload)}")
    else:
        print("WARNING: Expected cache key NOT found. The adapter will attempt to run SSL pretraining.")

    # 7. Check label budget for (seed, fraction)
    training_snaps = prepared.training_snapshots
    combined_x = torch.cat([s.x for s in training_snaps])
    combined_labels = torch.cat([s.labels for s in training_snaps])
    combined_ids = [tx for s in training_snaps for tx in s.tx_ids]
    combined_edges = training_snaps[0].edge_index if len(training_snaps) == 1 else torch.empty((2, 0), dtype=torch.long)
    exec_dataset = GraphData(combined_ids, combined_x, combined_labels, combined_edges)
    known_labels = {tx: int(label) for tx, label in zip(combined_ids, combined_labels.tolist()) if label in {0, 1}}

    budget = build_label_budgets(known_labels, seeds=(parsed.seed,), fractions=(parsed.fraction,))[(parsed.seed, parsed.fraction)]
    fit_labels = [known_labels[tx] for tx in budget.fit_ids]
    val_labels = [known_labels[tx] for tx in budget.validation_ids]
    refit_labels = [known_labels[tx] for tx in budget.refit_ids]
    print(f"\nLabel budget for seed={parsed.seed}, fraction={parsed.fraction}:")
    print(f"  Fit size        : {len(fit_labels)} (positives: {sum(fit_labels)}, negatives: {len(fit_labels) - sum(fit_labels)})")
    print(f"  Validation size : {len(val_labels)} (positives: {sum(val_labels)}, negatives: {len(val_labels) - sum(val_labels)})")
    print(f"  Refit size      : {len(refit_labels)} (positives: {sum(refit_labels)}, negatives: {len(refit_labels) - sum(refit_labels)})")

    # 8. Check existing matrix progress.json
    matrix_dir = args.artifacts_root / "matrices" / "s003-matrix-002"
    if matrix_dir.is_dir():
        prog_file = matrix_dir / "progress.json"
        if prog_file.is_file():
            print(f"\nExisting matrix progress ({prog_file}):")
            print(prog_file.read_text())

    if args.check_cache_only:
        print("\n[--check-cache-only] Diagnostic checks completed without training.")
        return 0

    if args.require_cache and cached_payload is None:
        print("\n[--require-cache] ABORTING: Cache key was not found. Halting before heavy SSL training.")
        return 1

    # 9. Execute cell via adapter directly to show exact traceback
    print("\n" + "=" * 80)
    print("STEP 1: Direct adapter.fit() Execution (Traceback Isolation)")
    print("=" * 80)

    adapter = create_adapter_for_cell(
        args.cell,
        ssl_epochs=ssl_epochs,
        downstream_epochs=downstream_epochs,
    )
    context = AdapterContext(
        seed=parsed.seed,
        fraction=parsed.fraction,
        fit_ids=frozenset(budget.fit_ids),
        validation_ids=frozenset(budget.validation_ids),
        refit_ids=frozenset(budget.refit_ids),
        budget_digest=budget.digest,
    )

    cached_embs = None
    cached_mod = None
    if cached_payload is not None:
        if isinstance(cached_payload, dict):
            cached_embs = cached_payload.get("embeddings")
            cached_mod = cached_payload.get("model")
        else:
            cached_embs = cached_payload

    try:
        import inspect
        sig = inspect.signature(adapter.fit)
        if "cached_embeddings" in sig.parameters:
            fitted = adapter.fit(
                exec_dataset,
                context,
                cached_embeddings=cached_embs,
                cached_model=cached_mod,
            )
        else:
            fitted = adapter.fit(exec_dataset, context)

        print("\n>>> SUCCESS: adapter.fit() completed without error!")
        downstream = getattr(fitted, "downstream", fitted)
        print(f"  Threshold       : {getattr(downstream, 'threshold', None)}")
        print(f"  Validation F1   : {getattr(downstream, 'validation_f1_illicit', None)}")
        print(f"  Validation MCC  : {getattr(downstream, 'validation_mcc', None)}")
        print(f"  Hyperparameters : {getattr(downstream, 'hyperparameters', None)}")

    except Exception as exc:
        print("\n>>> CAUGHT EXCEPTION in adapter.fit():")
        print(f"Exception Type    : {type(exc).__name__}")
        print(f"Exception Message : {exc}")
        print("\nFull Traceback:")
        traceback.print_exc()

    # 10. Execute via execute_matrix_cell to test scheduler handling
    print("\n" + "=" * 80)
    print("STEP 2: execute_matrix_cell() Execution (Scheduler State Test)")
    print("=" * 80)

    scheduler = MatrixScheduler(
        [args.cell],
        design_digest="0" * 64,
        test_labels=prepared.test_labels,
        expected_count=1,
    )

    cell_state = execute_matrix_cell(
        args.cell,
        scheduler,
        dataset=exec_dataset,
        budget=budget,
        data_digest=prepared.data_digest,
        pretraining_config_digest=config.digest,
        code_revision=effective_revision,
        config_digest=config.digest,
        embedding_cache=cache_store,
        ssl_epochs=ssl_epochs,
        downstream_epochs=downstream_epochs,
    )

    print(f"Cell final state   : {cell_state.state}")
    print(f"Failure kind       : {cell_state.failure_kind}")
    print(f"Failure message    : {cell_state.failure_message}")
    print(f"Weights digest     : {cell_state.weights_digest}")
    print(f"Threshold          : {cell_state.threshold}")

    print("\n" + "=" * 80)
    print("Diagnostic Complete.")
    print("=" * 80)
    return 0 if cell_state.state == "selected" else 2


if __name__ == "__main__":
    sys.exit(main())
