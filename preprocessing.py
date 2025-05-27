# IMPORT NECESSARY LIBRARIES ----------------------------------------------------------
import json
import spacy
from collections import defaultdict
import re
import pandas as pd

# DEFINE SOME FUNCTIONS TO PREPROCESS DATA ----------------------------------------------------------

# Load Spanish language model
nlp = spacy.load("es_core_news_sm")

def preprocess_data(json_list):
    """
    Preprocess the JSON data in the provided format.
    
    Args:
        json_list: List of dictionaries with "data" keys containing records
        
    Returns:
        A list of processed documents with features and labels
    """
    processed_data = []
    
    for record in json_list:
        # Extract text and annotations from each record
        text = record["data"]["text"]
        
        # If annotations are empty, use predictions
        annotations = record["predictions"]

        # Process annotations to extract cues and scopes
        neg_cues = []
        neg_scopes = []
        unc_cues = []
        unc_scopes = []
        
        for ann in annotations:
            if "result" in ann:
                for res in ann["result"]:
                    if res["type"] == "labels":
                        label = res["value"]["labels"][0]
                        start = res["value"]["start"]
                        end = res["value"]["end"]
                        span = (start, end)
                        
                        if label == "NEG":
                            neg_cues.append(span)
                        elif label == "NSCO":
                            neg_scopes.append(span)
                        elif label == "UNC":
                            unc_cues.append(span)
                        elif label == "USCO":
                            unc_scopes.append(span)
            
        # Process text with spaCy
        doc = nlp(text)
        doc_data = {
            "doc_id": record["data"]["id"],
            "text": text,
            "sentences": []
        }
        
        # For each sentence in the document
        for sent in doc.sents:
            sent_data = {
                "tokens": [],
                "features": [],
                "neg_cue_labels": [],
                "neg_scope_labels": [],
                "unc_cue_labels": [],
                "unc_scope_labels": []
            }
            
            # For each token in the sentence
            for token in sent:
                # Basic token features
                features = {
                    "word": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "shape": token.shape_,
                    "is_punct": token.is_punct,
                    "is_redacted": bool(re.match(r'^\*+$', token.text)),
                    "prefix": token.text[:3] if len(token.text) >= 3 else token.text,
                    "suffix": token.text[-3:] if len(token.text) >= 3 else token.text
                }
                
                # Check if token is a negation cue or in scope
                token_start = token.idx
                token_end = token.idx + len(token)

                # Use overlap condition for all labels
                def overlaps(start, end):
                    return start < token_end and end > token_start

                is_neg_cue = any(overlaps(start, end) for start, end in neg_cues)
                is_in_neg_scope = any(overlaps(start, end) for start, end in neg_scopes)
                is_unc_cue = any(overlaps(start, end) for start, end in unc_cues)
                is_in_unc_scope = any(overlaps(start, end) for start, end in unc_scopes)

                sent_data["tokens"].append(token)
                sent_data["features"].append(features)
                sent_data["neg_cue_labels"].append(1 if is_neg_cue else 0)
                sent_data["neg_scope_labels"].append(1 if is_in_neg_scope else 0)
                sent_data["unc_cue_labels"].append(1 if is_unc_cue else 0)
                sent_data["unc_scope_labels"].append(1 if is_in_unc_scope else 0)
            
            doc_data["sentences"].append(sent_data)
        
        processed_data.append(doc_data)
    
    return processed_data

def remove_false_negation_cues(processed_data, tokens_to_remove=None):
    """
    Set negation cue and scope labels to 0 for specific unwanted tokens (e.g. '.', ',', 'de', 'del').

    Args:
        processed_data: List of processed documents (from preprocess_data)
        tokens_to_remove: Set or list of token texts to reset labels for
    """
    if tokens_to_remove is None:
        tokens_to_remove = {'.', ',', ':', ';', 'de', 'del'}

    for doc in processed_data:
        for sent in doc["sentences"]:
            for i, token in enumerate(sent["tokens"]):
                if token.text.lower() in tokens_to_remove:
                    sent["neg_cue_labels"][i] = 0


def extract_lexicons(processed_data):
    """
    Extract lexicons of known cues from the processed data, including negation and uncertainty cues.
    
    Args:
        processed_data: The output from preprocess_data()
        
    Returns:
        Dictionaries of single-word cues and affixal cues
    """
    single_word_cues = set()
    affixal_cues = set()
    
    # Extract cues for negation and uncertainty from the processed data
    neg_cues = set()
    unc_cues = set()
    
    # Iterate over all the documents and sentences to collect cues
    for doc in processed_data:
        for sent in doc["sentences"]:
            # Add negation cues to the lexicon
            for token, is_neg_cue in zip(sent["tokens"], sent["neg_cue_labels"]):
                if is_neg_cue:
                    neg_cues.add(token.text.lower())
            
            # Add uncertainty cues to the lexicon
            for token, is_unc_cue in zip(sent["tokens"], sent["unc_cue_labels"]):
                if is_unc_cue:
                    unc_cues.add(token.text.lower())
                
            # Check for affixal cues (prefixes: a, anti, des, in, im; suffix: ment)
            for token in sent["tokens"]:
                # Check if token is a negation cue or in scope
                if token.text.lower() in neg_cues or token.text.lower() in unc_cues:
                    single_word_cues.add(token.text.lower())
                if token.text.lower().startswith(('a', 'anti', 'des', 'in', 'im')):
                    affixal_cues.add(token.text[:2].lower())  # Add the prefix
                elif token.text.lower().endswith('ment'):
                    affixal_cues.add('ment')
    
    # Return lexicons, including negation and uncertainty cues, as well as affixal cues
    return {
        "single_word_cues": list(single_word_cues),
        "affixal_cues": list(affixal_cues),
        "neg_cues": list(neg_cues),
        "unc_cues": list(unc_cues)
    }

def prepare_sequence_data_for_models(processed_data, lexicons, mode="negation"):
    """
    Prepare features and labels for SVM (cue detection) and CRF (scope detection).

    Args:
        processed_data: Output from preprocess_data()
        lexicons: Lexicons dictionary (from extract_lexicons)
        mode: Either 'negation' or 'uncertainty' to toggle which labels to use

    Returns:
        svm_features: List of per-sentence token features for SVM
        svm_labels: List of per-sentence labels (cues) for SVM
        crf_features: List of per-sentence token features for CRF
        crf_labels: List of per-sentence labels (scopes) for CRF
    """
    assert mode in {"negation", "uncertainty"}, "mode must be 'negation' or 'uncertainty'"

    svm_features = []
    svm_labels = []
    crf_features_list = []
    crf_labels_list = []

    cue_label_key = "neg_cue_labels" if mode == "negation" else "unc_cue_labels"
    scope_label_key = "neg_scope_labels" if mode == "negation" else "unc_scope_labels"

    for doc in processed_data:
        for sent in doc["sentences"]:
            tokens = sent["tokens"]
            cue_labels = sent[cue_label_key]
            scope_labels = sent[scope_label_key]

            sentence_svm_features = []
            sentence_svm_labels = []
            sentence_crf_features = []
            sentence_crf_labels = []

            for token, cue_label, scope_label in zip(tokens, cue_labels, scope_labels):
                # Common token features
                word_lower = token.text.lower()
                features_common = {
                    "word": word_lower,
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                    "prefix": word_lower[:3],
                    "suffix": word_lower[-3:],
                    "is_punct": int(token.is_punct),
                    "is_redacted": int(bool(re.match(r'^\*+$', token.text))),
                    "dep": token.dep_,
                    "head_pos": token.head.pos_,
                    "in_single_word_cues": int(word_lower in lexicons["single_word_cues"]),
                    "in_affixal_cues": int(any(word_lower.startswith(prefix) for prefix in lexicons["affixal_cues"])),
                    "ends_with_ment": int(word_lower.endswith("ment")),
                }

                # SVM (cue detection)
                sentence_svm_features.append(features_common)
                sentence_svm_labels.append(cue_label)

                # CRF (scope detection)
                sentence_crf_features.append(features_common)
                sentence_crf_labels.append(scope_label)

            svm_features.append(sentence_svm_features)
            svm_labels.append(sentence_svm_labels)
            crf_features_list.append(sentence_crf_features)
            crf_labels_list.append(sentence_crf_labels)

    return svm_features, svm_labels, crf_features_list, crf_labels_list


def prepare_lstm_data(processed_data, label_type="neg_cue_labels"):
    """
    Converts processed_data into token-label pairs for BiLSTM training.

    Args:
        processed_data: List of documents (from preprocess_data)
        label_type: One of 'neg_cue_labels', 'neg_scope_labels', 'unc_cue_labels', 'unc_scope_labels'

    Returns:
        List of (tokens, labels) tuples per sentence
    """
    lstm_data = []

    for doc in processed_data:
        for sent in doc["sentences"]:
            tokens = [token.text for token in sent["tokens"]]
            labels = sent[label_type]
            lstm_data.append((tokens, labels))

    return lstm_data


# OBTAIN AND PREPROCESS DATA ----------------------------------------------------------

with open('negacio_train_v2024.json') as f:
    json_train_data = json.load(f)

with open('negacio_test_v2024.json') as f:
    json_test_data = json.load(f)

# Process the data
processed_train_data = preprocess_data(json_train_data)
processed_test_data = preprocess_data(json_test_data)

remove_false_negation_cues(processed_train_data)
remove_false_negation_cues(processed_test_data)

# Extract lexicons
lexicons_train = extract_lexicons(processed_train_data)
lexicons_test = extract_lexicons(processed_test_data)

# Prepare sequence data and labels for the different models

# Train features and labels for -negations-
train_svm_feats_neg, train_svm_labels_neg, train_crf_feats_neg, train_crf_labels_neg = prepare_sequence_data_for_models(
    processed_train_data, lexicons_train, mode="negation")

# Train features and labels for -uncertainties-
train_svm_feats_unc, train_svm_labels_unc, train_crf_feats_unc, train_crf_labels_unc = prepare_sequence_data_for_models(
    processed_train_data, lexicons_train, mode="uncertainty")

# Tests features and labels for -negations-
test_svm_feats_neg, test_svm_labels_neg, test_crf_feats_neg, test_crf_labels_neg = prepare_sequence_data_for_models(
    processed_test_data, lexicons_test, mode="negation")

# Tests features and labels for -uncertainties-
test_svm_feats_unc, test_svm_labels_unc, test_crf_feats_unc, test_crf_labels_unc = prepare_sequence_data_for_models(
    processed_test_data, lexicons_test, mode="uncertainty")

# Train data (negation) for LSTM
lstm_train_data_neg_cue = prepare_lstm_data(processed_train_data, "neg_cue_labels")
lstm_train_data_neg_scope = prepare_lstm_data(processed_train_data, "neg_scope_labels")
# Test data (negation) for LSTM
lstm_test_data_neg_cue = prepare_lstm_data(processed_test_data, "neg_cue_labels")
lstm_test_data_neg_scope = prepare_lstm_data(processed_test_data, "neg_scope_labels")
# Train data (uncertainty) for LSTM
lstm_train_data_unc_cue = prepare_lstm_data(processed_train_data, "unc_cue_labels")
lstm_train_data_unc_scope = prepare_lstm_data(processed_train_data, "unc_scope_labels")
# Test data (uncertainty) for LSTM
lstm_test_data_unc_cue = prepare_lstm_data(processed_test_data, "unc_cue_labels")
lstm_test_data_unc_scope = prepare_lstm_data(processed_test_data, "unc_scope_labels")

# SAVE DATA INTO FILES ----------------------------------------------------------
import pickle

data_dict = {
    "lstm_train_data_neg_cue": lstm_train_data_neg_cue,
    "lstm_train_data_neg_scope": lstm_train_data_neg_scope,
    "lstm_test_data_neg_cue": lstm_test_data_neg_cue,
    "lstm_test_data_neg_scope": lstm_test_data_neg_scope,
    "lstm_train_data_unc_cue": lstm_train_data_unc_cue,
    "lstm_train_data_unc_scope": lstm_train_data_unc_scope,
    "lstm_test_data_unc_cue": lstm_test_data_unc_cue,
    "lstm_test_data_unc_scope": lstm_test_data_unc_scope,
}

with open("lstm_data.pkl", "wb") as f:
    pickle.dump(data_dict, f)


# CONVERT DATA INTO DATAFRAMES ---------------------------------------------------

def features_to_dataframe(features_list, labels_list=None, label_name="label"):

    # Flatten features and add sentence/token IDs
    flat_features = [token_feats for sent in features_list for token_feats in sent]
    df = pd.DataFrame(flat_features)
    
    # Add sentence and token identifiers
    df["sentence_id"] = [i for i, sent in enumerate(features_list) for _ in sent]
    df["token_id"] = [j for sent in features_list for j in range(len(sent))]
    
    # Add labels if provided
    if labels_list is not None:
        flat_labels = [label for sent in labels_list for label in sent]
        df[label_name] = flat_labels
    
    # Reorder columns to put IDs first
    columns = ["sentence_id", "token_id"] + [col for col in df.columns if col not in ["sentence_id", "token_id", label_name]]
    if labels_list is not None:
        columns.append(label_name)
    
    return df[columns]

# ---------SVM NEGATIONS------------

# Train data
df_svm_neg_train = features_to_dataframe(
    features_list=train_svm_feats_neg,
    labels_list=train_svm_labels_neg,
    label_name="neg_cue_label"  
)

print(df_svm_neg_train.head())
print("---------------------------------------")
print("\n")

# Test data
df_svm_neg_test = features_to_dataframe(
    features_list=test_svm_feats_neg,
    labels_list=test_svm_labels_neg,
    label_name="neg_cue_label"  
)

print(df_svm_neg_test.head())
print("---------------------------------------")
print("\n")

# ---------SVM UNCERTAINTIES------------

# Train data
df_svm_unc_train = features_to_dataframe(
    features_list=train_svm_feats_unc,
    labels_list=train_svm_labels_unc,
    label_name="unc_cue_label"  
)

print(df_svm_unc_train.head())
print("---------------------------------------")
print("\n")

# Test data
df_svm_unc_test = features_to_dataframe(
    features_list=test_svm_feats_unc,
    labels_list=test_svm_labels_unc,
    label_name="unc_cue_label"  
)

print(df_svm_unc_test.head())
print("---------------------------------------")
print("\n")

# ---------CRF NEGATIONS----------------

# Train data
df_crf_neg_train = features_to_dataframe(
    features_list=train_crf_feats_neg,
    labels_list=train_crf_labels_neg,
    label_name="neg_scope_label"  
)

print(df_crf_neg_train.head())
print("---------------------------------------")
print("\n")

# Test data
df_crf_neg_test = features_to_dataframe(
    features_list=test_crf_feats_neg,
    labels_list=test_crf_labels_neg,
    label_name="neg_scope_label"  
)

print(df_crf_neg_test)
print("---------------------------------------")
print("\n")

# ---------CRF UNCERTAINTIES------------

# Train data
df_crf_unc_train = features_to_dataframe(
    features_list=train_crf_feats_unc,
    labels_list=train_crf_labels_unc,
    label_name="unc_scope_label"
)

print(df_crf_unc_train.head())
print("---------------------------------------")
print("\n")

# Test data
df_crf_unc_test = features_to_dataframe(
    features_list=test_crf_feats_unc,
    labels_list=test_crf_labels_unc,
    label_name="unc_scope_label"
)

print(df_crf_unc_test.head())
print("---------------------------------------")
print("\n")

# Save DataFrame to a CSV file
"""
df_crf_unc_test.to_csv('df_unc_scopes_test.csv', index=False)
df_crf_unc_train.to_csv('df_unc_scopes_train.csv', index=False)
df_crf_neg_train.to_csv('df_neg_scopes_train.csv', index=False)
df_crf_neg_test.to_csv('df_neg_scopes_test.csv', index=False)
df_svm_neg_train.to_csv('df_neg_cues_train.csv', index=False)
df_svm_unc_train.to_csv('df_unc_cues_train.csv', index=False)
df_svm_neg_test.to_csv('df_neg_cues_test.csv', index=False)
df_svm_unc_test.to_csv('df_unc_cues_test.csv', index=False)
"""


# VISUALIZE DATA ---------------------------------------------------
def print_sample_sentence(processed_data, doc_index=0, sent_index=0):
    """
    Print tokens and their corresponding negation/uncertainty cue and scope labels for a specific sentence.

    Args:
        processed_data: List of preprocessed documents
        doc_index: Index of the document to inspect
        sent_index: Index of the sentence within the document
    """
    sent = processed_data[doc_index]["sentences"][sent_index]
    print(f"Document ID: {processed_data[doc_index]['doc_id']}")
    print(f"Sentence {sent_index + 1}:\n{'-'*50}")
    print(f"{'WORD':<15} {'NEG_CUE':<8} {'NEG_SCOPE':<10} {'UNC_CUE':<8} {'UNC_SCOPE'}")
    print(f"{'-'*50}")
    
    for tok, nc, ns, uc, us in zip(
        sent["tokens"], 
        sent["neg_cue_labels"],
        sent["neg_scope_labels"],
        sent["unc_cue_labels"],
        sent["unc_scope_labels"]
    ):
        print(f"{tok.text:<15} {nc:<8} {ns:<10} {uc:<8} {us}")

print_sample_sentence(processed_train_data, doc_index=1, sent_index=2)
