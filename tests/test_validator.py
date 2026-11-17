
from sequence_validator import validate_dna


def test_valid_dna_sequence():
    result, message = validate_dna("ATGC")
    assert result is True
    assert message == "DNA sequence is valid"


def test_lowercase_dna_sequence():
    result, message = validate_dna("atgc")
    assert result is True
    assert message == "DNA sequence is valid"


def test_empty_sequence():
    result, message = validate_dna("")
    assert result is False
    assert message == "Sequence cannot be empty"


def test_invalid_character():
    result, message = validate_dna("ATGX")
    assert result is False
    assert message == "Invalid DNA character: X"


def test_multiple_invalid_characters():
    result, message = validate_dna("ATGXYZ")
    assert result is False
    assert message == "Invalid DNA character: X"


def test_sequence_with_numbers():
    result, message = validate_dna("ATG123")
    assert result is False
    assert message == "Invalid DNA character: 1"


def test_sequence_with_spaces():
    result, message = validate_dna("AT GC")
    assert result is False
    assert message == "Invalid DNA character:  "
