from __future__ import annotations

import pytest

from hgcl.studies.s003.artifacts import EmbeddingCacheKey
from hgcl.studies.s003.config import ALL_FRACTIONS, ALL_METHODS, ALL_SEEDS, load_config
from hgcl.studies.s003.matrix import (
    CanonicalCellKey,
    MatrixScheduleError,
    build_dry_run_plans,
    build_matrix_coverage,
    canonical_matrix_design,
    cell_uses_embedding_cache,
    embedding_cache_key_for_cell,
    parse_cell_key,
)
from hgcl.studies.s003.pipeline import run_structural_dry_run


def test_canonical_design_has_205_p1_and_optional_five_p2_cells() -> None:
    p1 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)
    assert p1["p1_count"] == len(p1["cells"]) == 205
    assert len(set(p1["cells"])) == 205
    p2 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS, include_reverse_p2=True)
    assert p2["p1_count"] == 205 and p2["p2_count"] == 5
    assert len(p2["cells"]) == len(set(p2["cells"])) == 210
    assert p1["design_digest"] != p2["design_digest"]


def test_canonical_design_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="205 unique cells"):
        canonical_matrix_design(ALL_METHODS[:3], ALL_FRACTIONS, ALL_SEEDS)
    with pytest.raises(ValueError, match="205 unique cells"):
        canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS[:2], ALL_SEEDS)


def test_parse_cell_key_on_canonical_families() -> None:
    # 1. Main cell (SSL)
    main_ssl = parse_cell_key("main:s003_txgcl:0.01:11")
    assert main_ssl.family == "main"
    assert main_ssl.method == "s003_txgcl"
    assert main_ssl.variant == "complete"
    assert main_ssl.fraction == 0.01
    assert main_ssl.seed == 11
    assert main_ssl.is_p1 is True
    assert main_ssl.uses_ssl is True
    assert main_ssl.uses_embedding_cache is True

    # 2. Main cell (non-SSL)
    main_tabular = parse_cell_key("main:mlp_x:0.05:23")
    assert main_tabular.family == "main"
    assert main_tabular.method == "mlp_x"
    assert main_tabular.fraction == 0.05
    assert main_tabular.seed == 23
    assert main_tabular.is_p1 is True
    assert main_tabular.uses_ssl is False
    assert main_tabular.uses_embedding_cache is False

    # 3. Representation ablation (H-only)
    rep = parse_cell_key("representation:h_only:0.01:37")
    assert rep.family == "representation"
    assert rep.method == "s003_txgcl"
    assert rep.variant == "h_only"
    assert rep.fraction == 0.01
    assert rep.seed == 37
    assert rep.is_p1 is True
    assert rep.uses_ssl is True
    assert rep.uses_embedding_cache is True

    # 4. Mechanism ablation
    abl = parse_cell_key("ablation:no_knn:0.01:53")
    assert abl.family == "ablation"
    assert abl.method == "s003_txgcl"
    assert abl.variant == "no_knn"
    assert abl.seed == 53
    assert abl.is_p1 is True
    assert abl.uses_ssl is True
    assert abl.uses_embedding_cache is True

    # 5. P2 optional reverse edges
    p2 = parse_cell_key("p2:reverse_edges:0.01:71")
    assert p2.family == "p2"
    assert p2.method == "s003_txgcl"
    assert p2.variant == "reverse_edges"
    assert p2.seed == 71
    assert p2.is_p1 is False
    assert p2.uses_ssl is True
    assert p2.uses_embedding_cache is True


def test_parse_cell_key_rejects_malformed_keys() -> None:
    with pytest.raises(MatrixScheduleError, match="invalid canonical cell key"):
        parse_cell_key("main:s003_txgcl:0.01")
    with pytest.raises(MatrixScheduleError, match="invalid canonical cell key"):
        parse_cell_key("main:s003_txgcl:0.01:11:extra")
    with pytest.raises(MatrixScheduleError, match="invalid fraction or seed"):
        parse_cell_key("main:s003_txgcl:not_a_fraction:11")
    with pytest.raises(MatrixScheduleError, match="invalid fraction or seed"):
        parse_cell_key("main:s003_txgcl:0.01:not_a_seed")
    with pytest.raises(MatrixScheduleError, match="unknown canonical cell family"):
        parse_cell_key("unknown_family:variant:0.01:11")


def test_cell_uses_embedding_cache() -> None:
    assert cell_uses_embedding_cache("main:s003_txgcl:0.01:11") is True
    assert cell_uses_embedding_cache("representation:h_only:0.01:11") is True
    assert cell_uses_embedding_cache("ablation:no_knn:0.01:11") is True
    assert cell_uses_embedding_cache("ablation:no_edge_dropout:0.01:11") is True
    assert cell_uses_embedding_cache("ablation:random_individual:0.01:11") is True
    assert cell_uses_embedding_cache("ablation:random_groups:0.01:11") is True
    assert cell_uses_embedding_cache("p2:reverse_edges:0.01:11") is True
    assert cell_uses_embedding_cache("main:inspection_l_dgi:0.01:11") is True
    assert cell_uses_embedding_cache("main:gcpal:0.01:11") is True

    # Supervised GNNs and tabular methods do not use SSL embedding cache
    assert cell_uses_embedding_cache("main:mlp_x:0.01:11") is False
    assert cell_uses_embedding_cache("main:random_forest:0.01:11") is False
    assert cell_uses_embedding_cache("main:xgboost:0.01:11") is False
    assert cell_uses_embedding_cache("main:gcn_supervised:0.01:11") is False
    assert cell_uses_embedding_cache("main:graphsage_supervised:0.01:11") is False
    assert cell_uses_embedding_cache("main:gin_supervised:0.01:11") is False


def test_embedding_cache_key_reuses_pretraining_across_fractions() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    k01 = embedding_cache_key_for_cell("main:s003_txgcl:0.01:11", **common)
    k05 = embedding_cache_key_for_cell("main:s003_txgcl:0.05:11", **common)
    k10 = embedding_cache_key_for_cell("main:s003_txgcl:0.1:11", **common)
    k100 = embedding_cache_key_for_cell("main:s003_txgcl:1:11", **common)

    assert k01.digest == k05.digest == k10.digest == k100.digest


def test_embedding_cache_key_h_only_shares_s003_main_pretraining() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    main_key = embedding_cache_key_for_cell("main:s003_txgcl:0.01:11", **common)
    h_only_key = embedding_cache_key_for_cell("representation:h_only:0.01:11", **common)

    assert main_key.digest == h_only_key.digest


def test_embedding_cache_key_inspection_l_and_gcpal_reuse_across_fractions() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    insp_01 = embedding_cache_key_for_cell("main:inspection_l_dgi:0.01:11", **common)
    insp_100 = embedding_cache_key_for_cell("main:inspection_l_dgi:1:11", **common)
    assert insp_01.digest == insp_100.digest

    gcpal_01 = embedding_cache_key_for_cell("main:gcpal:0.01:11", **common)
    gcpal_100 = embedding_cache_key_for_cell("main:gcpal:1:11", **common)
    assert gcpal_01.digest == gcpal_100.digest

    assert insp_01.digest != gcpal_01.digest


def test_embedding_cache_key_ablations_are_distinct() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    main_key = embedding_cache_key_for_cell("main:s003_txgcl:0.01:11", **common)
    no_knn = embedding_cache_key_for_cell("ablation:no_knn:0.01:11", **common)
    no_edge = embedding_cache_key_for_cell("ablation:no_edge_dropout:0.01:11", **common)
    rand_ind = embedding_cache_key_for_cell("ablation:random_individual:0.01:11", **common)
    rand_grp = embedding_cache_key_for_cell("ablation:random_groups:0.01:11", **common)
    rev_edges = embedding_cache_key_for_cell("p2:reverse_edges:0.01:11", **common)

    digests = {k.digest for k in (main_key, no_knn, no_edge, rand_ind, rand_grp, rev_edges)}
    assert len(digests) == 6


def test_embedding_cache_key_distinct_across_seeds() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    digests = {
        embedding_cache_key_for_cell(f"main:s003_txgcl:0.01:{seed}", **common).digest
        for seed in ALL_SEEDS
    }
    assert len(digests) == 5


def test_embedding_cache_key_rejects_non_ssl_cells() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    with pytest.raises(MatrixScheduleError, match="does not use embedding cache"):
        embedding_cache_key_for_cell("main:mlp_x:0.01:11", **common)
    with pytest.raises(MatrixScheduleError, match="does not use embedding cache"):
        embedding_cache_key_for_cell("main:random_forest:0.01:11", **common)
    with pytest.raises(MatrixScheduleError, match="does not use embedding cache"):
        embedding_cache_key_for_cell("main:gcn_supervised:0.01:11", **common)


def test_canonical_p1_matrix_has_exactly_35_unique_ssl_pretrainings() -> None:
    """The 205-cell P1 matrix requires exactly 35 SSL pretrainings across 5 seeds."""
    p1 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}

    ssl_cells = [cell for cell in p1["cells"] if cell_uses_embedding_cache(cell)]
    assert len(ssl_cells) == 85  # (1 main * 4 frac + 1 rep + 4 abl) * 5 seeds + (2 ssl baselines * 4 frac * 5 seeds) = 45 + 40 = 85

    unique_keys = {embedding_cache_key_for_cell(cell, **common).digest for cell in ssl_cells}
    assert len(unique_keys) == 35  # Exactly 7 configurations x 5 seeds!

    # With P2 (reverse edges), exactly 40 SSL pretrainings
    p2 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS, include_reverse_p2=True)
    ssl_cells_p2 = [cell for cell in p2["cells"] if cell_uses_embedding_cache(cell)]
    unique_keys_p2 = {embedding_cache_key_for_cell(cell, **common).digest for cell in ssl_cells_p2}
    assert len(unique_keys_p2) == 40


def test_build_matrix_coverage_sc_001_and_sc_005() -> None:
    p1 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)
    coverage = build_matrix_coverage(p1["cells"])

    assert coverage["total_cells"] == 205
    assert coverage["expected_p1_cells"] == 205
    assert coverage["sc_001_compliant"] is True
    assert coverage["counts_by_state"] == {"planned": 205}
    assert coverage["counts_by_family"] == {
        "main": 180,
        "representation": 5,
        "ablation": 20,
    }

    # SC-005 verification in 1% regime
    sc005 = coverage["ablation_coverage_1pct"]
    for key, expected_len in [
        ("x_only", 5),
        ("h_only", 5),
        ("h_concat_x", 5),
        ("no_knn", 5),
        ("no_edge_dropout", 5),
        ("random_individual", 5),
        ("random_groups", 5),
        ("functional_blocks", 5),
    ]:
        assert len(sc005[key]) == expected_len, f"SC-005 missing {key}"


def test_build_dry_run_plans_structural_contents() -> None:
    config = load_config("configs/s003/dry-run.yaml")
    design = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)

    plans = build_dry_run_plans(config, design)

    # Coverage plan
    assert plans["coverage_plan"]["expected_p1_cells"] == 205
    assert plans["coverage_plan"]["total_cells"] == 205
    assert plans["coverage_plan"]["sc_001_compliant"] is True

    # Cache plan
    assert plans["cache_plan"]["unique_ssl_pretrainings"] == 35
    assert plans["cache_plan"]["ssl_reuse_across_fractions"] is True
    assert plans["cache_plan"]["h_only_reuses_main_ssl"] is True

    # Checkpoint plan
    assert plans["checkpoint_plan"]["atomic"] is True
    assert plans["checkpoint_plan"]["identity"] == "config-data-code"

    # Resume plan
    assert plans["resume_plan"]["allowed_state"] == "interrupted"
    assert plans["resume_plan"]["terminal_states_immutable"] is True

    # Resource plan
    assert plans["resource_plan"]["max_vram_gib"] == 4.5
    assert plans["resource_plan"]["max_ram_gib"] == 24
    assert plans["resource_plan"]["snapshot_batch_size"] == 1
    assert plans["resource_plan"]["projection_margin"] == 0.2

    # Duration projection without smoke timings
    assert plans["duration_projection"]["status"] == "pending_smoke_measurement"
    assert plans["duration_projection"]["projection_margin"] == 0.2

    # Duration projection with smoke timings (including 20% margin)
    smoke_timings = {"ssl_seconds": 10.0, "downstream_seconds": 2.0, "eval_seconds": 1.0}
    plans_with_timings = build_dry_run_plans(config, design, smoke_timings=smoke_timings)
    proj = plans_with_timings["duration_projection"]
    assert proj["status"] == "estimated"
    # base = 35 * 10 + 205 * 3 = 350 + 615 = 965
    assert proj["estimated_base_seconds"] == 965.0
    # total = 965 * 1.20 = 1158.0
    assert proj["estimated_total_seconds"] == 1158.0
