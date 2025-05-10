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

def extract_negations_and_uncertainties(data):
    """
    Extracts all unique negation ('NEG') and uncertainty ('UNC') spans from a DataFrame.
    Parameters:
    - df: Pandas DataFrame where each row contains 'data' and 'predictions' columns.

    Returns:
    - negations: List of unique negation text spans.
    - uncertainties: List of unique uncertainty text spans.
    """
    negations = set()
    uncertainties = set()

    for _, row in data.iterrows():
        # Safely get text and predictions
        text = row.get("data", {}).get("text", "")
        predictions = row.get("predictions", [])

        for pred in predictions:
            if not isinstance(pred, dict):
                continue

            for annotation in pred.get("result", []):
                if not isinstance(annotation, dict):
                    continue

                if annotation.get("type") == "labels":
                    labels = annotation.get("value", {}).get("labels", [])
                    if not labels:
                        continue

                    label = labels[0]
                    start = annotation.get("value", {}).get("start", 0)
                    end = annotation.get("value", {}).get("end", 0)
                    span = text[start:end].strip()

                    if label == "NEG":
                        negations.add(span)
                    elif label == "UNC":
                        uncertainties.add(span)

    return list(negations), list(uncertainties)