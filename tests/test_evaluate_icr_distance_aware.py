from scripts.evaluate_icr_distance_aware import advance_gate


def profiles(*, primary_correct=176, rescues=3, harms=2, improved=2, worsened=1,
             secondary_correct=175):
    return {
        "incumbent_general2": {"correct": 175},
        "primary_replace_general_seed7": {
            "correct": primary_correct, "rescues": rescues, "harms": harms,
            "folds_improved": improved, "folds_worsened": worsened,
        },
        "secondary_general2_plus_distance_seed7": {"correct": secondary_correct},
    }


def test_advance_gate_requires_every_preregistered_check():
    assert advance_gate(profiles())["advance_seed19"] is True
    assert advance_gate(profiles(primary_correct=175))["advance_seed19"] is False
    assert advance_gate(profiles(rescues=2, harms=2))["advance_seed19"] is False
    assert advance_gate(profiles(improved=1))["advance_seed19"] is False
    assert advance_gate(profiles(worsened=2))["advance_seed19"] is False
    assert advance_gate(profiles(secondary_correct=174))["advance_seed19"] is False
