
"""
GeneWeaver - Genomic FASTA Parser and Chunking Engine

Features:
- Parse FASTA files using BioPython
- Validate DNA sequences
- Divide long sequences into manageable chunks
- Preserve sequence identifiers and genomic positions
- Generate processing statistics
"""

from Bio import SeqIO
from pathlib import Path


VALID_BASES = set("ATGCN")


def validate_genome_sequence(sequence):
    """Validate a genomic sequence containing A, T, G, C and N."""

    sequence = str(sequence).upper().replace(" ", "").replace("\n", "")

    if not sequence:
        return False, "Empty sequence"

    invalid = sorted(set(sequence) - VALID_BASES)

    if invalid:
        return False, f"Invalid characters: {invalid}"

    return True, "Valid DNA sequence"


def parse_fasta_file(file_path):
    """Read FASTA records one at a time using BioPython."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    if path.suffix.lower() not in {".fasta", ".fa", ".fna"}:
        raise ValueError("Supported formats: .fasta, .fa and .fna")

    records_found = 0

    for record in SeqIO.parse(str(path), "fasta"):

        sequence = str(record.seq).upper()

        valid, message = validate_genome_sequence(sequence)

        if not valid:
            raise ValueError(
                f"Invalid sequence in record {record.id}: {message}"
            )

        records_found += 1

        yield {
            "id": record.id,
            "description": record.description,
            "sequence": sequence,
            "length": len(sequence)
        }

    if records_found == 0:
        raise ValueError("No FASTA records found in the file")


def split_sequence_into_chunks(
    sequence,
    chunk_size=100000,
    overlap=0
):
    """
    Split a DNA sequence into chunks.

    Positions are zero-based and end-exclusive.

    Example:
    chunk_size=4, overlap=1

    ATGCGTAC
    ATGC
       CGTA
          TAC
    """

    sequence = str(sequence).upper()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    valid, message = validate_genome_sequence(sequence)

    if not valid:
        raise ValueError(message)

    step = chunk_size - overlap

    for start in range(0, len(sequence), step):

        end = min(start + chunk_size, len(sequence))

        yield {
            "start": start,
            "end": end,
            "sequence": sequence[start:end],
            "length": end - start
        }

        if end == len(sequence):
            break


def process_fasta_in_chunks(
    file_path,
    chunk_size=100000,
    overlap=0
):
    """
    Parse FASTA records and generate chunks.

    This processes one FASTA record at a time and yields
    one chunk at a time instead of storing every chunk.
    """

    total_records = 0
    total_bases = 0
    total_chunks = 0

    for record in parse_fasta_file(file_path):

        total_records += 1
        total_bases += record["length"]

        for chunk_number, chunk in enumerate(
            split_sequence_into_chunks(
                record["sequence"],
                chunk_size,
                overlap
            ),
            start=1
        ):

            total_chunks += 1

            yield {
                "record_id": record["id"],
                "chunk_number": chunk_number,
                "start": chunk["start"],
                "end": chunk["end"],
                "length": chunk["length"],
                "sequence": chunk["sequence"]
            }

    # Summary is available through get_fasta_summary() below.
    # This generator intentionally avoids retaining all chunks.


def get_fasta_summary(file_path):
    """Calculate FASTA record and base-pair statistics."""

    record_count = 0
    total_bases = 0
    gc_bases = 0

    for record in parse_fasta_file(file_path):

        sequence = record["sequence"]

        record_count += 1
        total_bases += len(sequence)
        gc_bases += sequence.count("G") + sequence.count("C")

    gc_percentage = (
        (gc_bases / total_bases) * 100
        if total_bases else 0
    )

    return {
        "file": str(file_path),
        "records": record_count,
        "total_bases": total_bases,
        "gc_percentage": round(gc_percentage, 2)
    }


if __name__ == "__main__":

    print("=" * 55)
    print(" GENEWEAVER FASTA PARSER AND CHUNKING ENGINE")
    print("=" * 55)

    print("\nThis module is ready to process FASTA files.")
    print("Supported formats: .fasta, .fa, .fna")
    print("Default chunk size: 100,000 base pairs")
    print("Default overlap: 0 base pairs")
