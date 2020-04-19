def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds