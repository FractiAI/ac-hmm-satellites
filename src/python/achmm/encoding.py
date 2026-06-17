"""Nucleotide encoding utilities."""

ALPHABET = "ACGT"
SYMBOL_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_SYMBOL = {i: c for i, c in enumerate(ALPHABET)}


def encode_sequence(seq: str) -> list[int]:
    out: list[int] = []
    for ch in seq.upper():
        if ch in SYMBOL_TO_IDX:
            out.append(SYMBOL_TO_IDX[ch])
        else:
            out.append(-1)
    return out


def decode_sequence(symbols: list[int]) -> str:
  return "".join(IDX_TO_SYMBOL.get(s, "N") for s in symbols)


def active_mask_from_sequence(seq: str) -> list[int]:
    return [1 if ch.upper() in SYMBOL_TO_IDX else 0 for ch in seq]
