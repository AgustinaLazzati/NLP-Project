# IMPORT NECESSARY LIBRARIES ----------------------------------------------------------
import json
import spacy
from collections import defaultdict
import re

# PREPROCESS DATA ----------------------------------------------------------

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
                token_end = token.idx + len(token.text)
                
                is_neg_cue = any(start <= token_start < end for start, end in neg_cues)
                is_in_neg_scope = any(start <= token_start < end for start, end in neg_scopes)
                is_unc_cue = any(start <= token_start < end for start, end in unc_cues)
                is_in_unc_scope = any(start <= token_start < end for start, end in unc_scopes)
                
                sent_data["tokens"].append(token.text)
                sent_data["features"].append(features)
                sent_data["neg_cue_labels"].append(1 if is_neg_cue else 0)
                sent_data["neg_scope_labels"].append(1 if is_in_neg_scope else 0)
                sent_data["unc_cue_labels"].append(1 if is_unc_cue else 0)
                sent_data["unc_scope_labels"].append(1 if is_in_unc_scope else 0)
            
            doc_data["sentences"].append(sent_data)
        
        processed_data.append(doc_data)
    
    return processed_data


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
                    neg_cues.add(token.lower())
            
            # Add uncertainty cues to the lexicon
            for token, is_unc_cue in zip(sent["tokens"], sent["unc_cue_labels"]):
                if is_unc_cue:
                    unc_cues.add(token.lower())
                
            # Check for affixal cues (prefixes: a, anti, des, in, im; suffix: ment)
            for token in sent["tokens"]:
                # Check if token is a negation cue or in scope
                if token.lower() in neg_cues or token.lower() in unc_cues:
                    single_word_cues.add(token.lower())
                if token.lower().startswith(('a', 'anti', 'des', 'in', 'im')):
                    affixal_cues.add(token[:2].lower())  # Add the prefix
                elif token.lower().endswith('ment'):
                    affixal_cues.add('ment')
    
    # Return lexicons, including negation and uncertainty cues, as well as affixal cues
    return {
        "single_word_cues": list(single_word_cues),
        "affixal_cues": list(affixal_cues),
        "neg_cues": list(neg_cues),
        "unc_cues": list(unc_cues)
    }

def prepare_sequence_data_for_models(processed_data, lexicons):
    """
    Prepare features and labels for both SVM and CRF models.
    
    Returns two sets of data:
    - SVM data for cue detection (negation and uncertainty).
    - CRF data for scope detection (negation and uncertainty scopes).
    """
    svm_features = []
    svm_labels = []
    crf_features = []
    crf_labels = []

    for doc in processed_data:
        for sent in doc["sentences"]:
            sentence_tokens = sent["tokens"]
            sentence_features = []
            sentence_crf_features = []
            sentence_svm_labels = []
            sentence_crf_labels = []
            
            for token, neg_cue_label, unc_cue_label, neg_scope_label, unc_scope_label in zip(
                    sent["tokens"], sent["neg_cue_labels"], sent["unc_cue_labels"],
                    sent["neg_scope_labels"], sent["unc_scope_labels"]):
                
                # Create features for SVM (cue detection)
                features = {
                    "word": token.lower(),
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                    "prefix": token.text[:3].lower(),
                    "suffix": token.text[-3:].lower(),
                    "is_punct": int(token.is_punct),
                    "is_redacted": int(bool(re.match(r'^\*+$', token.text))),
                    "dep": token.dep_,
                    "head_pos": token.head.pos_,
                }

                # Add lexicon-based features for SVM
                features["in_single_word_cues"] = int(token.lower() in lexicons["single_word_cues"])
                features["in_affixal_cues"] = int(any(token.lower().startswith(prefix) for prefix in lexicons["affixal_cues"]))
                features["ends_with_ment"] = int(token.lower().endswith("ment"))
                
                sentence_features.append(features)
                sentence_svm_labels.append(neg_cue_label)  # APPEND `unc_cue_label` FOR UNCERTAINTY CUE DETECTION
                
                # Create features for CRF (scope detection)
                crf_features = {
                    "word": token.lower(),
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                    "prefix": token.text[:3].lower(),
                    "suffix": token.text[-3:].lower(),
                    "is_punct": int(token.is_punct),
                    "is_redacted": int(bool(re.match(r'^\*+$', token.text))),
                    # Add dependency features
                    "dep": token.dep_,
                    "head_pos": token.head.pos_,
                }

                # Add lexicon-based features for CRF
                crf_features["in_single_word_cues"] = int(token.lower() in lexicons["single_word_cues"])
                crf_features["in_affixal_cues"] = int(any(token.lower().startswith(prefix) for prefix in lexicons["affixal_cues"]))
                crf_features["ends_with_ment"] = int(token.lower().endswith("ment"))
                
                sentence_crf_features.append(crf_features)
                sentence_crf_labels.append(neg_scope_label)  # APPEND `unc_scope_label` FOR UNCERTAINTY SCOPE DETECTION

            # Append sentence features and labels to document-level lists
            svm_features.append(sentence_features)
            svm_labels.append(sentence_svm_labels)
            crf_features.append(sentence_crf_features)
            crf_labels.append(sentence_crf_labels)

    return svm_features, svm_labels, crf_features, crf_labels

# MAIN ----------------------------------------------------------

with open('negacio_train_v2024.json') as f:
    json_train_data = json.load(f)

with open('negacio_test_v2024.json') as f:
    json_test_data = json.load(f)

# Process the data
processed_train_data = preprocess_data(json_train_data)
processed_test_data = preprocess_data(json_test_data)

# Extract lexicons
lexicons_train = extract_lexicons(processed_train_data)
lexicons_test = extract_lexicons(processed_test_data)

# Prepare sequence data and labels for the different models
train_features_svm, train_labels_svm, train_features_crf, train_labels_crf = prepare_sequence_data_for_models(processed_train_data, lexicons_train)
test_features_svm, test_labels_svm, test_features_crf, test_features_crf = prepare_sequence_data_for_models(processed_test_data, lexicons_test)
