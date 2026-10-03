from app.rag.citations import (
    extract_source_numbers,
    validate_source_numbers,
)


def test_extract_source_numbers():
    answer = (
        "The artist discusses creation [1][3]. "
        "Another character mentions sculpture [2]. "
        "Source [1] also supports this."
    )

    result = extract_source_numbers(answer)

    assert result == [1, 3, 2]


def test_empty_answer():
    assert extract_source_numbers("") == []


def test_invalid_source_numbers():
    answer = "This is supported by [1][3][9]."

    valid, invalid = validate_source_numbers(
        answer,
        source_count=5,
    )

    assert valid == [1, 3]
    assert invalid == [9]


def test_all_valid():
    answer = "Evidence appears in [1][2][5]."

    valid, invalid = validate_source_numbers(
        answer,
        source_count=5,
    )

    assert valid == [1, 2, 5]
    assert invalid == []


def test_no_citations():
    valid, invalid = validate_source_numbers(
        "There is not enough information.",
        source_count=5,
    )

    assert valid == []
    assert invalid == []


print("Testing citation extraction...")

test_extract_source_numbers()
test_empty_answer()
test_invalid_source_numbers()
test_all_valid()
test_no_citations()

print("ALL CITATION TESTS PASSED")