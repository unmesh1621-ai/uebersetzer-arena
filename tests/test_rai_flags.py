from src.rai_probe import flags


def test_number_mismatch():
    assert "number_mismatch" in flags("I have 3 cats.", "Ich habe 4 Katzen.")


def test_untranslated():
    assert "untranslated" in flags("Hello world, this is a sentence.", "Hello world, this is a sentence.")


def test_repetition():
    assert "repetition" in flags("Hi.", "ja ja ja ja ja ja ja ja")


def test_length_ratio():
    assert "length_ratio" in flags("Yes.", "Ja, das ist ganz sicher richtig und ich stimme dem vollkommen zu.")


def test_clean_translation():
    assert flags("Good morning, how are you today?", "Guten Morgen, wie geht es dir heute?") == []


def test_rerun_keeps_labels_for_unchanged_outputs():
    import pandas as pd
    from src.rai_probe import merge_labels

    old = pd.DataFrame([{"source": "a", "system": "mt5", "output": "x", "back_translation": "bt", "label": "ok", "note": "n"},
                        {"source": "b", "system": "mt5", "output": "y", "back_translation": "bt2", "label": "major", "note": ""}])
    new = pd.DataFrame([{"source": "a", "system": "mt5", "output": "x", "back_translation": "", "label": "", "note": ""},
                        {"source": "b", "system": "mt5", "output": "CHANGED", "back_translation": "", "label": "", "note": ""}])
    out = merge_labels(new, old)
    assert out.loc[0, ["back_translation", "label", "note"]].tolist() == ["bt", "ok", "n"]
    assert out.loc[1, "label"] == ""  # output changed -> must be relabelled
    assert out.output.tolist() == ["x", "CHANGED"]
