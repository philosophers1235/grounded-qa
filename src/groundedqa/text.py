"""Tokenising, stemming and sentence splitting shared by indexing and answering."""

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Standard English stopwords plus words that only shape a question ("how long",
# "what happens", "do I need"). Leaving those in makes a well-covered question
# look unsupported because the documents rarely use them.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing done down during
    each either else ever few for from further get gets got had has have having he her
    here hers herself him himself his how i if in into is it its itself just let me
    might more most must my myself no nor not now of off on once only or other our ours
    ourselves out over own please same shall she should so some such tell than that the
    their theirs them themselves then there these they this those through to too under
    until up upon us very via was way we were what when where whether which while who
    whom why will with within without would yes yet you your yours yourself yourselves
    able anything happen happens happened know long many much need needs onto someone
    something want wants
    """.split()
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")


def stem(word: str) -> str:
    """Strip common English suffixes so 'installed' and 'installation' meet at 'install'.

    Deliberately small: it only has to be consistent between documents and queries.
    """
    if len(word) <= 3 or word[0].isdigit():
        return word

    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("sses", "ches", "shes", "xes", "zes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]

    for suffix in ("ation", "ment", "ing", "ed", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
            break

    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS or (len(raw) == 1 and not raw.isdigit()):
            continue
        tokens.append(stem(raw))
    return tokens


def query_terms(text: str) -> list[str]:
    """Unique tokens in first-seen order."""
    return list(dict.fromkeys(tokenize(text)))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences. Each line is split on its own, so list items stay separate."""
    spans = []
    offset = 0
    for line in text.split("\n"):
        start = 0
        for match in _SENTENCE_END.finditer(line):
            spans.append((offset + start, offset + match.start()))
            start = match.end()
        spans.append((offset + start, offset + len(line)))
        offset += len(line) + 1

    trimmed = []
    for start, end in spans:
        segment = text[start:end]
        lead = len(segment) - len(segment.lstrip())
        trail = len(segment) - len(segment.rstrip())
        if end - trail > start + lead:
            trimmed.append((start + lead, end - trail))
    return trimmed


def strip_list_marker(sentence: str) -> str:
    return _LIST_MARKER.sub("", sentence, count=1)
