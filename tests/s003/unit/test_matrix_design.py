from hgcl.studies.s003.artifacts import EmbeddingCacheKey
from hgcl.studies.s003.config import ALL_FRACTIONS, ALL_METHODS, ALL_SEEDS
from hgcl.studies.s003.pipeline import canonical_matrix_design


def test_canonical_design_has_205_p1_and_optional_five_p2_cells() -> None:
    p1 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)
    assert p1["p1_count"] == len(p1["cells"]) == 205
    assert len(set(p1["cells"])) == 205
    p2 = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS, include_reverse_p2=True)
    assert p2["p1_count"] == 205 and p2["p2_count"] == 5
    assert len(p2["cells"]) == len(set(p2["cells"])) == 210
    assert p1["design_digest"] != p2["design_digest"]


def test_embedding_cache_key_reuses_pretraining_across_fractions() -> None:
    key = EmbeddingCacheKey("s003_txgcl", "complete", 11, "a" * 64, "b" * 64, "c" * 40)
    same = EmbeddingCacheKey("s003_txgcl", "complete", 11, "a" * 64, "b" * 64, "c" * 40)
    other_seed = EmbeddingCacheKey("s003_txgcl", "complete", 23, "a" * 64, "b" * 64, "c" * 40)
    assert key.digest == same.digest
    assert key.digest != other_seed.digest
