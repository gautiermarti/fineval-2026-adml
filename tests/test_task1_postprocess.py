import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.task1_postprocess import (
    add_explicit_identifiers, add_high_support_nested_locations, add_zone_designators,
)


def test_exact_athens_exchange_phrase_adds_nested_location_once():
    entities = [("Χρηµατιστήριο Αθηνών", "ΟΡΓΑΝΙΣΜΟΣ")]
    assert add_high_support_nested_locations(entities) == [
        ("Χρηµατιστήριο Αθηνών", "ΟΡΓΑΝΙΣΜΟΣ"),
        ("Αθηνών", "ΤΟΠΟΘΕΣΙΑ"),
    ]


def test_other_athens_organization_does_not_trigger_nested_rule():
    entities = [("Τράπεζα Αθηνών", "ΟΡΓΑΝΙΣΜΟΣ")]
    assert add_high_support_nested_locations(entities) == entities


def test_train_supported_country_genitives_are_nested_locations():
    entities = [
        ("Vodafone Ελλάδας", "ΟΡΓΑΝΙΣΜΟΣ"),
        ("INFORM LYKOS Ρουμανίας", "ΟΡΓΑΝΙΣΜΟΣ"),
    ]
    assert add_high_support_nested_locations(entities) == entities + [
        ("Ελλάδας", "ΤΟΠΟΘΕΣΙΑ"),
        ("Ρουμανίας", "ΤΟΠΟΘΕΣΙΑ"),
    ]

def test_explicit_gemi_afm_and_auditor_ids():
    text='Α.Μ. Σ.Ο.Ε.Λ. 26911, ΓΕΜΗ 3518601000 και Α.Φ.Μ. 095674330.'
    assert add_explicit_identifiers(text, []) == [
        ('26911','ΑΛΛΑ'), ('3518601000','ΑΛΛΑ'), ('095674330','ΑΛΛΑ')
    ]

def test_spaced_gemi_and_identifier_fragments():
    text='αρ. ΓΕ.ΜΗ 3518601000 και Α.Φ.Μ. 095674330.'
    entities=[('74','ΠΟΣΟΤΗΤΕΣ'),('33','ΧΡΗΜΑΤΑ')]
    assert add_explicit_identifiers(text, entities) == [
        ('3518601000','ΑΛΛΑ'), ('095674330','ΑΛΛΑ')
    ]

def test_repeated_number_outside_identifier_is_retained():
    text='Η χρήση 91 αναφέρεται πριν από τον Α.Μ. Σ.Ο.Ε.Λ. 26911.'
    assert add_explicit_identifiers(text, [('91','ΠΟΣΟΤΗΤΕΣ')]) == [
        ('91','ΠΟΣΟΤΗΤΕΣ'), ('26911','ΑΛΛΑ')
    ]

def test_calendar_day_before_greek_month_is_temporal():
    text='Η συναλλαγή ολοκληρώθηκε την 31η Μαΐου 2021.'
    assert add_explicit_identifiers(text, [('2021','ΧΡΟΝΙΚΑ')]) == [
        ('2021','ΧΡΟΝΙΚΑ'), ('31','ΧΡΟΝΙΚΑ')
    ]


def test_inflected_calendar_day_suffix_is_replaced_by_numeric_core():
    text = 'Η αποκοπή ορίστηκε για την 8ης Αυγούστου 2023.'
    once = add_explicit_identifiers(text, [
        ('8ης', 'ΧΡΟΝΙΚΑ'), ('2023', 'ΧΡΟΝΙΚΑ'), ('8', 'ΧΡΟΝΙΚΑ'),
    ])
    assert once == [('2023', 'ΧΡΟΝΙΚΑ'), ('8', 'ΧΡΟΝΙΚΑ')]
    assert add_explicit_identifiers(text, once) == once


def test_inflected_calendar_day_eta_suffix_is_replaced_by_numeric_core_once():
    text = 'Η συνέλευση θα γίνει την 20η Μαΐου.'
    assert add_explicit_identifiers(text, [('20η', 'ΧΡΟΝΙΚΑ')]) == [
        ('20', 'ΧΡΟΝΙΚΑ')
    ]


def test_suffix_like_temporal_surface_outside_calendar_syntax_is_retained():
    text = 'Η αναφορά στην 8ης χρήσης παραμένει ως είχε.'
    assert add_explicit_identifiers(text, [('8ης', 'ΧΡΟΝΙΚΑ')]) == [
        ('8ης', 'ΧΡΟΝΙΚΑ')
    ]


def test_inflected_calendar_surface_with_another_label_is_retained():
    text = 'Η συνέλευση θα γίνει την 20η Μαΐου.'
    assert add_explicit_identifiers(text, [('20η', 'ΑΛΛΑ')]) == [
        ('20η', 'ΑΛΛΑ'), ('20', 'ΧΡΟΝΙΚΑ')
    ]


def test_ambiguous_repeated_inflected_surface_is_retained_conservatively():
    text = 'Η 8ης χρήσης έκλεισε πριν από την 8ης Αυγούστου.'
    assert add_explicit_identifiers(text, [('8ης', 'ΧΡΟΝΙΚΑ')]) == [
        ('8ης', 'ΧΡΟΝΙΚΑ'), ('8', 'ΧΡΟΝΙΚΑ')
    ]


def test_labeled_legal_designators_are_other_numbers():
    text='Σύμφωνα με το άρθρο 21 παρ. 3 του Ν. 4209/2013 και τη Σημείωση 15.'
    assert add_explicit_identifiers(text, [('2013','ΧΡΟΝΙΚΑ')]) == [
        ('2013','ΧΡΟΝΙΚΑ'), ('21','ΑΛΛΑ'), ('3','ΑΛΛΑ'),
        ('15','ΑΛΛΑ'), ('4209','ΑΛΛΑ'),
    ]


def test_law_rule_does_not_match_ordinary_greek_word_ending_in_nu():
    text='Η μεταφορά των 100% θυγατρικών ολοκληρώθηκε.'
    assert add_explicit_identifiers(text, [('100%','ΠΟΣΟΣΤΑ')]) == [
        ('100%','ΠΟΣΟΣΤΑ')
    ]


def test_zone_designator_is_optional_other_number():
    text='Το έργο καλύπτει τη Ζώνη 2 και τη ζώνη 5.'
    assert add_zone_designators(text, []) == [('2','ΑΛΛΑ'), ('5','ΑΛΛΑ')]


def test_zone_rule_does_not_split_decimal_quantity():
    text='Η χερσαία ζώνη 39.400 τ.μ. παραμένει αμετάβλητη.'
    assert add_zone_designators(text, []) == []
