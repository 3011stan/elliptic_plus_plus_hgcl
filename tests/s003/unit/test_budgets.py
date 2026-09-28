from __future__ import annotations

from hgcl.studies.s003.splits import build_label_budgets


def test_nested_stratified_budgets_all_seeds_and_fractions() -> None:
    labels = {f"i{i:03d}": 1 for i in range(200)} | {f"l{i:03d}": 0 for i in range(200)} | {"unknown": -1}
    budgets = build_label_budgets(labels)
    assert len(budgets) == 20
    for seed in (11, 23, 37, 53, 71):
        chain = [budgets[(seed, fraction)] for fraction in (0.01, 0.05, 0.10, 1.0)]
        for budget in chain:
            assert budget.fit_ids.isdisjoint(budget.validation_ids)
            assert budget.refit_ids == budget.fit_ids | budget.validation_ids
            assert set(labels[i] for i in budget.fit_ids) == {0, 1}
            assert set(labels[i] for i in budget.validation_ids) == {0, 1}
        for smaller, larger in zip(chain, chain[1:]):
            assert smaller.fit_ids <= larger.fit_ids
            assert smaller.validation_ids <= larger.validation_ids
    assert budgets[(11, 0.01)].digest == build_label_budgets(labels)[(11, 0.01)].digest

