
from pam_detection import find_pam_sites


def test_single_pam_site():
    result = find_pam_sites("ATGAGG")
    assert result == [
        {"position": 3, "pam": "AGG"}
    ]


def test_multiple_pam_sites():
    result = find_pam_sites("AGGATGG")
    assert result == [
        {"position": 0, "pam": "AGG"},
        {"position": 4, "pam": "TGG"}
    ]


def test_no_pam_site():
    result = find_pam_sites("ATGCTA")
    assert result == []


def test_lowercase_sequence():
    result = find_pam_sites("atgagg")
    assert result == [
        {"position": 3, "pam": "AGG"}
    ]


def test_empty_sequence():
    result = find_pam_sites("")
    assert result == "DNA sequence cannot be empty"


def test_invalid_sequence():
    result = find_pam_sites("ATGXGG")
    assert result == "Invalid DNA sequence"


def test_sequence_shorter_than_pam():
    result = find_pam_sites("AG")
    assert result == []
