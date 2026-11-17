
# GeneWeaver - DNA Sequence Validator

def validate_dna(sequence):

    sequence = sequence.upper()

    valid_bases = {"A", "T", "G", "C"}

    if not sequence:
        return False, "Sequence cannot be empty"

    for base in sequence:
        if base not in valid_bases:
            return False, f"Invalid DNA character: {base}"

    return True, "DNA sequence is valid"



# Test multiple DNA sequences

test_sequences = [
    "ATGCGTACGTAG",
    "ATGCXYZ",
    "ATG123",
    "",
    "atgcgt"
]

print("\nGeneWeaver - DNA Validation Tests")
print("---------------------------------")

for sequence in test_sequences:

    is_valid, message = validate_dna(sequence)

    print("\nSequence:", sequence)

    if is_valid:
        print("Result: VALID")
    else:
        print("Result: INVALID")

    print("Message:", message)

