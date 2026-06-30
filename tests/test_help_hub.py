from core.help_content import HELP_RECIPES, CATEGORY_ORDER

# Ids that external code deep-links to; must never change.
DEEPLINK_IDS = {
    "one_way_anova", "two_way_anova", "repeated_measures_anova", "ancova",
}

def _by_id():
    return {r["id"]: r for r in HELP_RECIPES}

def test_every_recipe_has_known_category():
    for r in HELP_RECIPES:
        assert "category" in r, f"{r['id']} missing category"
        assert r["category"] in CATEGORY_ORDER, f"{r['id']} has unknown category {r['category']!r}"

def test_required_keys_present():
    for r in HELP_RECIPES:
        for key in ("id", "title", "summary", "keywords", "html", "category"):
            assert key in r and r[key], f"{r['id']} has empty or missing {key}"

def test_ids_unique():
    ids = [r["id"] for r in HELP_RECIPES]
    assert len(ids) == len(set(ids))

def test_deeplink_ids_preserved():
    ids = set(_by_id())
    assert DEEPLINK_IDS <= ids

def test_migrated_recipes_present():
    ids = set(_by_id())
    assert {"dependent_samples", "graph_visualization", "statistical_tests_html"} <= ids

def test_category_order_is_valid():
    assert len(CATEGORY_ORDER) > 0
    assert len(CATEGORY_ORDER) == len(set(CATEGORY_ORDER))
