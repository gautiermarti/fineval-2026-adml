import sys
from pathlib import Path

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))

from scripts.train_task1_transformer import align_annotations


def test_alignment_does_not_require_answer_order():
    text='Στις 31.12.2017 το ποσοστό ήταν 97,0%.'
    spans=[('97,0%','ΠΟΣΟΣΤΑ'),('2017','ΧΡΟΝΙΚΑ'),('12','ΧΡΟΝΙΚΑ'),('31','ΧΡΟΝΙΚΑ')]
    assert align_annotations(text,spans) == [
        (5,7,'ΧΡΟΝΙΚΑ'),(8,10,'ΧΡΟΝΙΚΑ'),(11,15,'ΧΡΟΝΙΚΑ'),
        (32,37,'ΠΟΣΟΣΤΑ'),
    ]


def test_alignment_uses_distinct_repeated_occurrences():
    text='στις 08.09.2017 και στις 08.12.2017'
    aligned=align_annotations(text,[
        ('08','ΧΡΟΝΙΚΑ'),('08','ΧΡΟΝΙΚΑ'),
        ('2017','ΧΡΟΝΙΚΑ'),('2017','ΧΡΟΝΙΚΑ'),
    ])
    assert [(text[left:right],label) for left,right,label in aligned] == [
        ('08','ΧΡΟΝΙΚΑ'),('2017','ΧΡΟΝΙΚΑ'),
        ('08','ΧΡΟΝΙΚΑ'),('2017','ΧΡΟΝΙΚΑ'),
    ]


def test_numeric_substring_is_not_aligned_inside_larger_number():
    assert align_annotations('κωδικός 26911',[('91','ΠΟΣΟΤΗΤΕΣ')]) == []
