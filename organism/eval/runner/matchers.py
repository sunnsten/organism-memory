import re
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from razdel import tokenize
    from pymorphy2 import MorphAnalyzer

try:
    from razdel import tokenize
    from pymorphy2 import MorphAnalyzer
    HAS_NLP_LIBS = True
except ImportError:
    HAS_NLP_LIBS = False

# Initialise the morphological analyser once at module level
_morph = None

def _get_morph():
    """Lazy initialisation of the morphological analyser."""
    global _morph
    if _morph is None and HAS_NLP_LIBS:
        try:
            _morph = MorphAnalyzer()
        except Exception:
            # pymorphy2 may be installed but broken (e.g. Python 3.13 removed inspect.getargspec)
            return None
    return _morph


def normalize(text: str) -> str:
    """
    Normalise text for comparison using razdel + pymorphy2:
    - tokenise with razdel
    - lemmatise with pymorphy2
    - lower-case
    - ё → е
    - collapse whitespace

    Falls back to simple lower-case + whitespace collapse when the NLP
    libraries are not installed.
    """
    # Normalise thousands separators before tokenisation so "1,300" and "1300" match.
    # Handles: 1,300 → 1300 and 1.300 → 1300 (German/EU style).
    text = re.sub(r'(\d)[,\.](\d{3})\b', r'\1\2', text)

    if not HAS_NLP_LIBS:
        text = text.lower()
        text = text.replace("ё", "е")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    tokens = list(tokenize(text))

    morph = _get_morph()
    if morph is None:
        words = [token.text.lower() for token in tokens]
    else:
        words = []
        for token in tokens:
            word = token.text.lower()
            word = word.replace("ё", "е")
            parsed_list = morph.parse(word)
            if parsed_list:
                parsed = parsed_list[0]
                # type: ignore[attr-defined] - pymorphy2 Parse object has normal_form
                words.append(parsed.normal_form)  # type: ignore[attr-defined]
            else:
                words.append(word)

    normalized = " ".join(words)
    normalized = " ".join(normalized.split())

    return normalized.strip()


def contains_all(text: str, tokens: List[str]) -> bool:
    """
    Return True if every token appears in text after normalisation.

    Args:
        text: text to search in
        tokens: tokens that must all be present
    """
    normalized_text = normalize(text)
    normalized_tokens = [normalize(token) for token in tokens]

    for token in normalized_tokens:
        if token not in normalized_text:
            return False

    return True


def contains_none(text: str, tokens: List[str]) -> bool:
    """
    Return True if none of the tokens appear in text after normalisation.

    Args:
        text: text to search in
        tokens: tokens that must all be absent
    """
    normalized_text = normalize(text)
    normalized_tokens = [normalize(token) for token in tokens]

    for token in normalized_tokens:
        if token in normalized_text:
            return False

    return True


def token_overlap_ratio(answer: str, snippet: str) -> float:
    """
    Return the token overlap ratio between answer and snippet after normalisation.

    Returns:
        Float in [0.0, 1.0]: |answer_tokens ∩ snippet_tokens| / |snippet_tokens|
    """
    normalized_answer = normalize(answer)
    normalized_snippet = normalize(snippet)

    answer_tokens = set(normalized_answer.split())
    snippet_tokens = set(normalized_snippet.split())

    if not snippet_tokens:
        return 0.0

    intersection = answer_tokens & snippet_tokens
    return len(intersection) / len(snippet_tokens)
