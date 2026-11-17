
import time

def needleman_wunsch(sequence1, sequence2,
                     match=2, mismatch=-1, gap=-2):

    rows = len(sequence1) + 1
    cols = len(sequence2) + 1

    # Create scoring matrix
    matrix = [
        [0 for _ in range(cols)]
        for _ in range(rows)
    ]

    # Initialize gap penalties
    for i in range(rows):
        matrix[i][0] = i * gap

    for j in range(cols):
        matrix[0][j] = j * gap

    # Fill scoring matrix
    for i in range(1, rows):
        for j in range(1, cols):

            if sequence1[i - 1] == sequence2[j - 1]:
                diagonal = matrix[i - 1][j - 1] + match
            else:
                diagonal = matrix[i - 1][j - 1] + mismatch

            up = matrix[i - 1][j] + gap
            left = matrix[i][j - 1] + gap

            matrix[i][j] = max(diagonal, up, left)

    # Traceback to construct aligned sequences
    aligned1 = []
    aligned2 = []

    i = len(sequence1)
    j = len(sequence2)

    while i > 0 or j > 0:

        if i > 0 and j > 0:
            score = match if sequence1[i - 1] == sequence2[j - 1] else mismatch

            if matrix[i][j] == matrix[i - 1][j - 1] + score:
                aligned1.append(sequence1[i - 1])
                aligned2.append(sequence2[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and matrix[i][j] == matrix[i - 1][j] + gap:
            aligned1.append(sequence1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(sequence2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()

    return "".join(aligned1), "".join(aligned2), matrix[-1][-1]


if __name__ == "__main__":

    dna1 = "ATGCGTAC"
    dna2 = "ATGAGTAC"

    print("=" * 50)
    print(" GENEWEAVER CPU GLOBAL ALIGNMENT")
    print("=" * 50)

    start = time.perf_counter()

    result1, result2, score = needleman_wunsch(dna1, dna2)

    elapsed = time.perf_counter() - start

    print("DNA Sequence 1:", dna1)
    print("DNA Sequence 2:", dna2)
    print("\nALIGNED SEQUENCES")
    print("Sequence 1:", result1)
    print("Sequence 2:", result2)
    print("Alignment score:", score)
    print(f"CPU execution time: {elapsed * 1000:.6f} ms")
