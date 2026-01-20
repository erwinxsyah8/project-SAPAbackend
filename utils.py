import re
from collections import Counter

def extract_keywords(text: str, top_n=5):
    words = re.findall(r"\b\w+\b", text.lower())
    common = Counter(words).most_common(top_n)
    return [w for w, _ in common]
