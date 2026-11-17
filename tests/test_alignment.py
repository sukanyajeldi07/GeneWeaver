
import pytest

from global_alignment_cpu import needleman_wunsch
from global_alignment_gpu import gpu_needleman_wunsch


@pytest.mark.parametrize(
    "dna1, dna2",
    [
        ("ATGCGTAC", "ATGCTAC"),
        ("ATGCGTAC", "ATGAGTAC"),
        ("AAAA", "AAAA"),
        ("ATGC", "GCAT"),
        ("A", "A"),
    ],
)
def test_cpu_gpu_alignment_match(dna1, dna2):

    cpu_result = needleman_wunsch(dna1, dna2)
    gpu_result = gpu_needleman_wunsch(dna1, dna2)

    assert cpu_result[0] == gpu_result[0]
    assert cpu_result[1] == gpu_result[1]
    assert cpu_result[2] == gpu_result[2]


def test_identical_sequences():

    result = needleman_wunsch("ATGC", "ATGC")

    assert result[0] == "ATGC"
    assert result[1] == "ATGC"
    assert result[2] == 8


def test_empty_sequences():

    result = needleman_wunsch("", "")

    assert result[0] == ""
    assert result[1] == ""
    assert result[2] == 0
