import numpy as np
import pandas as pd
import requests
from readability import Document
import dataset

def get_text_from_url(url):
    '''
    Get most significant text returned from URL.
    '''
    response = requests.get(url)
    doc = Document(response.text)
    text = dataset.preprocess_html(doc.summary())
    return text


def predict_labels(clf, labels, text, top_k=3):
    '''
    Returns top_k labels for text.
    '''
    vectorizer = clf.named_steps['tfidf']
    tfidf = vectorizer.transform([text])
    probs = clf.named_steps['clf'].predict_proba(tfidf)
    
    preds = np.argsort(probs, axis=1)[0,-top_k:][::-1]
    probs = probs[:, preds][0]
    pred_labels = list(map(lambda x: labels[labels.CLASS == x].TEMA.values[0], preds))
    preds_df = pd.DataFrame({'LABEL': pred_labels, 'PROBABILITY': probs})
    
    #if verbose:
    print(preds_df, '\n')
    return preds_df


def predict_labels_url(clf, labels, url, top_k=3, verbose=False):
    text = get_text_from_url(url)
    if verbose:
        print('Text:\n\n{}\n'.format(text))
    return predict_labels(clf, labels, text, top_k), text


def top_tfidf_feats(row, features, top_k=10):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_k]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['TOP TERMS', 'TF_IDF SCORE']
    return df


def top_terms(clf, text, top_k=10):
    ''' Top tfidf features in specific document (matrix row) '''
    
    vectorizer = clf.named_steps['tfidf']
    tfidf = vectorizer.transform([text])    
    row = np.squeeze(tfidf.toarray())
    return top_tfidf_feats(row, vectorizer.get_feature_names(), top_k)


def top_terms_url(clf, url, top_k=10, verbose=False):
    text = get_text_from_url(url)
    if verbose:
        print('Text:\n\n{}\n'.format(text))
    return top_terms(clf, text, top_k), text