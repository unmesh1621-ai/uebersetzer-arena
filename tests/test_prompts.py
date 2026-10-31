from src.prompts import INSTRUCTION, build_messages, postprocess


def test_build_messages_with_shots():
    msgs = build_messages("Hi", shots=[("a", "b")])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == INSTRUCTION.format(src="a")
    assert msgs[1]["content"] == "b"
    assert msgs[-1]["content"] == INSTRUCTION.format(src="Hi")


def test_build_messages_with_target():
    msgs = build_messages("Hi", tgt="Hallo")
    assert len(msgs) == 2
    assert msgs[-1] == {"role": "assistant", "content": "Hallo"}


def test_postprocess_strips_label_and_extra_lines():
    assert postprocess("German: Guten Tag.\n\nNote: this is formal.") == ("Guten Tag.", True)


def test_postprocess_strips_quotes():
    assert postprocess('"Hallo Welt"') == ("Hallo Welt", True)


def test_postprocess_clean_untouched():
    assert postprocess("Hallo") == ("Hallo", False)


def test_postprocess_empty():
    assert postprocess("") == ("", False)


def test_postprocess_leading_blank_lines():
    assert postprocess("\n\nGuten Abend.") == ("Guten Abend.", True)
