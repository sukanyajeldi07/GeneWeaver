
# GeneWeaver - CRISPR Guide Matching

def find_guide_matches(dna_sequence, guide_sequence):

    dna_sequence = dna_sequence.upper()
    guide_sequence = guide_sequence.upper()

    valid_bases = {"A", "T", "G", "C"}

    if not dna_sequence or not guide_sequence:
        return "DNA and guide sequences cannot be empty"

    if not set(dna_sequence).issubset(valid_bases):
        return "Invalid DNA sequence"

    if not set(guide_sequence).issubset(valid_bases):
        return "Invalid guide sequence"

    matches = []

    guide_length = len(guide_sequence)

    for i in range(len(dna_sequence) - guide_length + 1):

        section = dna_sequence[i:i + guide_length]

        if section == guide_sequence:
            matches.append(i)

    return matches


# Sample test
dna = "ATGCGTACGTAGCTAG"
guide = "CGTA"

result = find_guide_matches(dna, guide)

print("\nGeneWeaver - CRISPR Guide Matching")
print("----------------------------------")
print("DNA Sequence:", dna)
print("Guide Sequence:", guide)

if isinstance(result, list):

    if result:
        print("Matching positions (0-based):", result)
        print("Total matches:", len(result))
    else:
        print("No exact matches found")

else:
    print("Error:", result)
