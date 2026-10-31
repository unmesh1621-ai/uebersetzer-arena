from src.data import clean_pairs

GOOD = ("Good morning, how are you today?", "Guten Morgen, wie geht es dir heute?")


def kept(pairs, blocklist=frozenset()):
    return clean_pairs(pairs, set(blocklist))[0]


def test_keeps_good_pair():
    assert kept([GOOD]) == [GOOD]


def test_drops_empty():
    out, stats = clean_pairs([("", "Hallo Welt, wie geht es?"), GOOD], set())
    assert out == [GOOD]
    assert stats["empty"] == 1


def test_drops_duplicate():
    out, stats = clean_pairs([GOOD, GOOD], set())
    assert out == [GOOD]
    assert stats["duplicate"] == 1


def test_drops_too_long():
    long_en = " ".join(["word"] * 101)
    long_de = " ".join(["Wort"] * 101)
    out, stats = clean_pairs([(long_en, long_de)], set())
    assert out == []
    assert stats["too_long"] == 1


def test_drops_bad_ratio():
    pair = ("one two", "eins zwei drei vier fünf sechs sieben acht")
    out, stats = clean_pairs([pair], set())
    assert out == []
    assert stats["ratio"] == 1


def test_drops_non_german_target():
    pair = ("This is a test sentence.", "This is clearly English text and not German at all.")
    out, stats = clean_pairs([pair], set())
    assert out == []
    assert stats["langid"] == 1


def test_drops_flores_leak():
    out, stats = clean_pairs([GOOD], {GOOD[0]})
    assert out == []
    assert stats["leak"] == 1


def test_nfc_normalises():
    pair = ("The green door is open.", "Die grüne Tür ist offen.")
    out = kept([pair])
    assert out[0][1] == "Die grüne Tür ist offen."
