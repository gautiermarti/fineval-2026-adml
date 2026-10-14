from fineval.icr_zunda import parse_zunda_block, split_zunda_blocks, zunda_signature


SAMPLE = """#FUNCEXP\tO,O,O,O,B:完了,O
#EVENT0\t3\twr:筆者\t非未来\t0\t叙述\t成立\t0\t0
* 0 -1D 0/1 0.0
投資\t名詞,サ変接続,*,*,*,*,投資,トウシ,トーシ
を\t助詞,格助詞,一般,*,*,*,を,ヲ,ヲ
実行\t名詞,サ変接続,*,*,*,*,実行,ジッコウ,ジッコー
した\t動詞,自立,*,*,サ変・スル,連用形,する,シ,シ
。\t記号,句点,*,*,*,*,。,。,。
EOS
"""


def test_parse_zunda_block_resolves_event_token():
    block = split_zunda_blocks(SAMPLE)[0]
    events, functional = parse_zunda_block(block)
    assert events[0].surface == "した"
    assert events[0].lemma == "する"
    assert events[0].actuality == "成立"
    assert functional[4] == "B:完了"


def test_zunda_signature_is_query_conditioned_and_normalized():
    candidate = {
        "start": 5, "input_only_score": 2.0, "question_overlap": 0.5,
        "ruri_similarity": 0.8,
        "future_cues": 1, "commit_cues": 1, "hedge_cues": 0,
        "refusal_cues": 0, "conditional_cues": 0,
        "functional_expressions": ["O", "B:判断"],
        "zunda_events": [events.to_dict() for events in parse_zunda_block(
            split_zunda_blocks(SAMPLE)[0]
        )[0]],
    }
    signature = zunda_signature([candidate])
    assert signature["global:actuality=成立"] == 1.0
    assert signature["query_weighted:actuality=成立"] == 1.0
    assert signature["top_input:question_overlap"] == 0.5
    assert signature["top_semantic:ruri_similarity"] == 0.8
    assert signature["semantic_weighted:actuality=成立"] == 1.0

