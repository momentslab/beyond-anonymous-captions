"""Post-processing for generated captions."""

import re


def clean_awkward_phrasings(text: str) -> str:
    """Remove awkward phrasings like 'the person named X'."""
    name_pat = r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)"

    text = re.sub(
        r"\b[Tt]he (?:man|woman|person|individual|figure|boy|girl|actor|actress|guy|lady|astronaut) named "
        + name_pat,
        r"\1",
        text,
    )
    text = re.sub(r"\b[Tt]he one named " + name_pat, r"\1", text)
    text = re.sub(
        r"\b[Aa] (?:man|woman|person|individual|figure|boy|girl) named " + name_pat,
        r"\1",
        text,
    )
    text = re.sub(
        r"\b[Aa] (?:man|woman|person|individual|figure|boy|girl) identified as "
        + name_pat
        + r",?",
        r"\1",
        text,
    )
    text = re.sub(r",?\s*identified as " + name_pat + r",?", r" \1", text)
    text = re.sub(r",?\s*who is named " + name_pat + r",?", r" \1", text)
    text = re.sub(r",?\s*known as " + name_pat + r",?", r" \1", text)
    text = re.sub(r",?\s*called " + name_pat + r",?", r" \1", text)
    text = re.sub(r",?\s*labeled as " + name_pat + r",?", r" \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
