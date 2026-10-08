"""Workload synthesis and quality datasets for IndicXlit evaluations."""

from __future__ import annotations

import random
from typing import Sequence

# Curated Roman-to-Hindi gold dataset for quick quality verification.
QUALITY_SAMPLES: list[tuple[str, str]] = [
    ("mera", "मेरा"),
    ("bharat", "भारत"),
    ("namaste", "नमस्ते"),
    ("nyayalaya", "न्यायालय"),
    ("adhivakta", "अधिवक्ता"),
    ("kanoon", "कानून"),
    ("kachahari", "कचहरी"),
    ("vakeel", "वकील"),
    ("insaaf", "इंसाफ"),
    ("pradhanmantri", "प्रधानमंत्री"),
    ("samvidhan", "संविधान"),
    ("vidhansabha", "विधानसभा"),
    ("adhikaran", "अधिकरण"),
    ("mukadma", "मुकदमा"),
    ("faisla", "फैसला"),
    ("gawahi", "गवाही"),
    ("praman", "प्रमाण"),
    ("muddai", "मुद्दई"),
    ("prativadi", "प्रतिवादी"),
    ("nyayadhish", "न्यायाधीश"),
    ("shasan", "शासन"),
    ("adhikar", "अधिकार"),
    ("sarkar", "सरकार"),
    ("police", "पुलिस"),
    ("daroga", "दरोगा"),
    ("thana", "थाना"),
    ("aaropi", "आरोपी"),
    ("saboot", "सबूत"),
    ("bayan", "बयान"),
    ("peshi", "पेशी"),
]

SMOKE_WORDS = ["mera", "bharat", "namaste", "nyayalaya", "adhivakta"]

SHORT = [
    "ram", "dil", "man", "sab", "kal", "aaj", "des", "mera", "naam", "desh",
    "raja", "prem", "kala", "naya", "baat", "ghar", "pyar", "hind", "nyay", "dafa",
]

MEDIUM = [
    "bharat", "kanoon", "vakeel", "kachahari", "insaaf", "namaste", "vakalat",
    "mukadma", "gawahi", "faisla", "praman", "muddai", "nyayalay", "samvidhan",
    "nyayalaya", "adhivakta", "prativadi", "nyayadhish", "adhikaran", "kanooniat",
]

LONG = [
    "nyayapalika", "vidhansabha", "manavadhikar", "panchayatraj", "pradhanmantri",
    "samvidhanik", "adhivaktagana", "adhikaranseva", "nyayalayadheesh",
    "uchchanyayalaya", "kanoonvyavastha", "adhivaktaseva", "nyayikprashasan",
    "nyayalayabhavan", "loksabhasadasya", "adhivaktaparishad", "nyayikpunriksha",
    "samvidhansabha", "antarashtreey", "nyayadhikaran",
]

CONSONANTS = [
    "k", "kh", "g", "gh", "ch", "j", "t", "th", "d", "dh",
    "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "s", "h",
]
VOWELS = ["a", "aa", "i", "ee", "u", "oo", "e", "ai", "o", "au"]

BUCKET_RANGES: dict[str, tuple[int, int]] = {
    "short": (3, 5),
    "medium": (6, 10),
    "long": (11, 18),
}


def generate_word(bucket: str, rng: random.Random) -> str:
    """
    Generate a single phonetically plausible Roman word of target length.
    """
    target_bucket = bucket
    if bucket == "mixed":
        target_bucket = rng.choices(["short", "medium", "long"], weights=[0.50, 0.35, 0.15])[0]
    min_len, max_len = BUCKET_RANGES[target_bucket]
    target_len = rng.randint(min_len, max_len)

    parts: list[str] = []
    curr_len = 0
    use_consonant = rng.choice([True, False])
    while curr_len < target_len:
        pool = CONSONANTS if use_consonant else VOWELS
        s = rng.choice(pool)
        parts.append(s)
        curr_len += len(s)
        use_consonant = not use_consonant
    return "".join(parts)[:target_len]


def generate_unique_word(bucket: str, seen: set[str], rng: random.Random) -> str:
    """
    Generate a guaranteed unique word not already in seen set.
    """
    for _ in range(100):
        word = generate_word(bucket, rng)
        if word not in seen:
            seen.add(word)
            return word
    word = f"{generate_word(bucket, rng)}{len(seen) + 1}"
    seen.add(word)
    return word


def generate_batch(
    batch_size: int,
    bucket: str,
    seen: set[str],
    rng: random.Random,
) -> list[str]:
    """
    Generate a batch of unique Roman words.
    """
    return [generate_unique_word(bucket, seen, rng) for _ in range(batch_size)]


def generate_workload(
    requests: int,
    batch_size: int,
    bucket: str,
    seed: int | None = None,
) -> tuple[list[list[str]], int]:
    """
    Pre-generate batches of unique words and return (batches, seed).
    """
    if seed is None:
        seed = random.randint(1000, 999999)
    rng = random.Random(seed)
    seen: set[str] = set()

    if bucket == "smoke":
        batches: list[list[str]] = []
        for i in range(requests):
            batch = [SMOKE_WORDS[(i * batch_size + j) % len(SMOKE_WORDS)] for j in range(batch_size)]
            batches.append(batch)
        return batches, seed

    batches = [generate_batch(batch_size, bucket, seen, rng) for _ in range(requests)]
    return batches, seed
