import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from html.parser import HTMLParser
from unidecode import unidecode

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def preprocess_html(html):
    text = strip_tags(html)
    return unidecode(text)


def get_most_similar(query, vecs, vocabulary=None, k=3):
    '''
    Get k most similar vectors using cosine similarity.
    '''
    cosine = cosine_similarity(query.reshape(1, -1), vecs)[0]
    max_idxs = np.array(cosine).argsort()[::-1]
    mask = np.in1d(max_idxs, list(vocabulary.keys()))
    max_idxs = max_idxs[mask][:k]
    return ['' if vocabulary.get(idx) == '</PAD>' else vocabulary.get(idx) for idx in max_idxs]


def filter_relevant(data, min_percent, class_column='CLASS', label_column='TEMA'):
    '''
    Filter data by removing classes that account for
    less than `min_percent` of the data. This is useful
    for dealing with unbalanced datasets.
    '''
    by_class = data.groupby(class_column, as_index=False)
    cols = data.columns.tolist()
    cols.remove(class_column)
    freqs = by_class.count().sort_values(cols[0])
    freqs['cum_percent'] = (freqs[cols[0]]/freqs[cols[0]].sum()).cumsum()
    significant_classes = freqs[freqs.cum_percent > min_percent][class_column]
    data_filtered = data[data[class_column].isin(significant_classes)].copy()
    data_filtered['ORIGINAL_' + class_column] = data_filtered[class_column]
    # reset class labels
    sorted_classes = np.sort(data_filtered[class_column].unique())
    data_filtered[class_column] = data_filtered[class_column].map(
        lambda x: np.where(sorted_classes == x)[0][0])

    data_other = data[(~data[class_column].isin(
        data_filtered['ORIGINAL_' + class_column]) & data[class_column].notnull())].copy()
    data_other['ORIGINAL_' + class_column] = data_other[class_column]
    data_other['ORIGINAL_' + label_column] = data_other[label_column]
    n_classes = len(data_filtered[class_column].unique())
    data_other[class_column] = data_other[class_column].map(lambda x: n_classes)
    data_other[label_column] = data_other[label_column].map(lambda x: 'Outros')
    data_filtered = data_filtered.append(data_other)

    return data_filtered


def get_excerpt(sentence, max_length=300, keep_first=100):
    '''
    Returns a random excerpt from a text.

    Parameters:
        sentence - a string containing the text
        max_length - max number of words in excerpt
        keep_first - number of words to preserve in
            the beginning of the sentence
    '''

    if isinstance(sentence, str):
        sentence = sentence.split()

    excerpt = sentence

    if len(sentence) > max_length:

        excerpt = sentence[:keep_first]
        excerpt_start = random.randint(keep_first, len(sentence)- 1)

        if len(sentence) > excerpt_start + max_length - keep_first:
            excerpt += sentence[excerpt_start:excerpt_start + max_length - keep_first]
        else:
            excerpt += sentence[excerpt_start:]

    return ' '.join(excerpt)


def augment(sentences, labels, aug_factor=5, max_length=300):
    '''
    Create augmented dataset using random excerpts.

    Parameters:
        sentences - a list of strings containing the sentences
        labels - a list containing the target class for each sentence
        aug_factor - augmentation factor
        max_length - max number of words in each augmented sample
    '''

    x = []
    y = []

    for index, sentence in enumerate(sentences):
        for i in range(aug_factor):
            excerpt = get_excerpt(sentence, max_length=max_length)
            x.append(excerpt)
            y.append(labels[index])

    return x, y