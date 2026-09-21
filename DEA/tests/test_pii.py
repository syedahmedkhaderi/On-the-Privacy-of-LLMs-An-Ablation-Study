from dea_a3 import pii


def test_email_extraction_and_normalisation():
    vals = pii.extract_values("write to Jane.Doe@Enron.COM today")
    assert vals["email"] == {"jane.doe@enron.com"}


def test_phone_formats_all_normalise_to_ten_digits():
    for text in ["713-853-6161", "(713) 853-6161", "713.853.6161",
                 "+1 713 853 6161", "7138536161"]:
        vals = pii.extract_values(text)
        assert vals["phone"] == {"7138536161"}, text


def test_phone_rejects_invalid_area_code():
    assert pii.extract_values("123-456-7890")["phone"] == set()
    assert pii.extract_values("013-456-7890")["phone"] == set()


def test_card_requires_luhn():
    assert pii.extract_values("4111111111111111")["card"] == {"4111111111111111"}
    assert pii.extract_values("4111111111111112")["card"] == set()


def test_card_not_double_counted_as_phone():
    kinds = {s.kind for s in pii.extract_spans("4111111111111111")}
    assert "card" in kinds and "phone" not in kinds


def test_email_blocks_overlapping_phone():
    assert [s.kind for s in pii.extract_spans("a7138536161@enron.com")] == ["email"]


def test_salted_hash_is_stable_and_typed():
    salt = "abc"
    h1 = pii.salted_hash("email", "A@B.com", salt)
    h2 = pii.salted_hash("email", "a@b.com", salt)
    assert h1 == h2
    assert pii.salted_hash("phone", "7138536161", salt) != \
           pii.salted_hash("card", "7138536161", salt)
    assert h1 != pii.salted_hash("email", "a@b.com", "different")


def test_mask_never_leaks_full_value():
    assert pii.mask("email", "jane.doe@enron.com") == "j*******@enron.com"
    assert pii.mask("phone", "7138536161") == "713XXXXXXX"


def test_empty_text_is_safe():
    assert pii.extract_spans("") == []
    assert pii.extract_values("")["email"] == set()
