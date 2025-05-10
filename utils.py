import re

def preprocess_text(text, keep_case=False):
    if not isinstance(text, str) or not text:
        return ""

    # Convert text to lowercase
    if keep_case:
        text = text.lower()

    # Remove asterisks
    text = re.sub(r'\*+', 'REDACTED', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text
