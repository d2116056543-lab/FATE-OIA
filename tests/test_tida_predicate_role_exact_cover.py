import yaml

from fate_oia.models.tida_predicate_differential import load_predicate_roles


def test_predicate_roles_cover_scene_names_exactly_once():
    scene = yaml.safe_load(open("configs/aie_scene_predicates.yaml", encoding="utf-8"))
    names = [row["name"] for row in scene["predicates"]]
    roles = load_predicate_roles("configs/tida_predicate_roles.yaml", names)
    flattened = [name for members in roles.values() for name in members]
    assert sorted(flattened) == sorted(names)
    assert len(flattened) == len(set(flattened)) == 32
