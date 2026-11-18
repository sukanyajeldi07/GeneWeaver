
from genome_parser import (
    validate_genome_sequence,
    split_sequence_into_chunks,
    parse_fasta_file,
    process_fasta_in_chunks,
    get_fasta_summary,
)


def test_valid_genome_sequence():
    valid, message = validate_genome_sequence("ATGCGTACNN")
    assert valid is True


def test_invalid_genome_sequence():
    valid, message = validate_genome_sequence("ATGCXYZ")
    assert valid is False


def test_empty_genome_sequence():
    valid, message = validate_genome_sequence("")
    assert valid is False


def test_sequence_chunking():
    chunks = list(
        split_sequence_into_chunks("ATGCGTAC", chunk_size=4)
    )

    assert len(chunks) == 2
    assert chunks[0]["sequence"] == "ATGC"
    assert chunks[1]["sequence"] == "GTAC"


def test_chunk_positions():
    chunks = list(
        split_sequence_into_chunks("ATGCGTAC", chunk_size=4)
    )

    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 4
    assert chunks[1]["start"] == 4
    assert chunks[1]["end"] == 8


def test_chunk_overlap():
    chunks = list(
        split_sequence_into_chunks(
            "ATGCGTAC",
            chunk_size=4,
            overlap=1
        )
    )

    assert chunks[0]["sequence"] == "ATGC"
    assert chunks[1]["sequence"] == "CGTA"
    assert chunks[2]["sequence"] == "AC"


def test_invalid_chunk_size():
    try:
        list(split_sequence_into_chunks("ATGC", chunk_size=0))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_overlap():
    try:
        list(
            split_sequence_into_chunks(
                "ATGC",
                chunk_size=4,
                overlap=4
            )
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_fasta_parser(tmp_path):
    fasta = tmp_path / "sample.fasta"

    fasta.write_text(
        ">sequence_1\nATGCGTAC\n"
        ">sequence_2\nGGCCAATT\n",
        encoding="utf-8"
    )

    records = list(parse_fasta_file(fasta))

    assert len(records) == 2
    assert records[0]["id"] == "sequence_1"
    assert records[0]["length"] == 8
    assert records[1]["sequence"] == "GGCCAATT"


def test_fasta_chunk_processing(tmp_path):
    fasta = tmp_path / "sample.fasta"

    fasta.write_text(
        ">sequence_1\nATGCGTAC\n",
        encoding="utf-8"
    )

    chunks = list(
        process_fasta_in_chunks(fasta, chunk_size=4)
    )

    assert len(chunks) == 2
    assert chunks[0]["sequence"] == "ATGC"
    assert chunks[1]["sequence"] == "GTAC"


def test_fasta_summary(tmp_path):
    fasta = tmp_path / "sample.fasta"

    fasta.write_text(
        ">sequence_1\nATGC\n"
        ">sequence_2\nGGCC\n",
        encoding="utf-8"
    )

    summary = get_fasta_summary(fasta)

    assert summary["records"] == 2
    assert summary["total_bases"] == 8
    assert summary["gc_percentage"] == 75.0
