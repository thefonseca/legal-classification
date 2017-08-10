import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from html.parser import HTMLParser
from unidecode import unidecode
from sklearn.externals import joblib
from urllib.request import urlretrieve
from tqdm import tqdm
import os, errno
import zipfile


def check_makedir(path):
    '''
    Creates a directory if it does not exist.
    '''
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def extract_zip(filename, dest_dir=None):
    '''
    Extracts a zip file to specified folder.
    '''

    print('Extracting {}...'.format(filename))

    if not dest_dir:
        dest_dir = os.path.dirname(filename)
    
    with zipfile.ZipFile(filename) as f:
        f.extractall(dest_dir)
        extracted_file = os.path.join(dest_dir, f.namelist()[0])
        print('File extracted to:', extracted_file)
        return extracted_file


def urlopen_with_progress(url, dest_filename):
    def my_hook(t):
        """
        Wraps tqdm instance. Don't forget to close() or __exit__() the tqdm instance
        once you're done (easiest using a context manager, eg: `with` syntax)

        Example
        -------

        >>> with tqdm(...) as t:
        ...     reporthook = my_hook(t)
        ...     urllib.urlretrieve(..., reporthook=reporthook)
        """
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            """
            b     : int, optional    Number of blocks just transferred [default: 1]
            bsize : int, optional    Size of each block (in tqdm units) [default: 1]
            tsize : int, optional    Total size (in tqdm units). If [default: None]
                                     remains unchanged.
            """
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner

    with tqdm(unit='B', unit_scale=True, miniters=1,
              desc="Downloading file...") as t:
        return urlretrieve(url, dest_filename, reporthook=my_hook(t))

    #with open(filename, 'r') as f:
    #    return f.read()


def maybe_download(url, filename, expected_bytes, force=False, data_root='.'):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    check_makedir(dest_filename)

    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        #filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        filename, _ = urlopen_with_progress(url.format(filename), dest_filename)
        print('\nDownload Complete!')

        statinfo = os.stat(dest_filename)
        
        if statinfo.st_size == expected_bytes:
            print('Found and verified', dest_filename)
        else:
            raise Exception(
                'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')

    elif os.path.exists(dest_filename):
        print('File already exists:', dest_filename)
        
    return dest_filename


def save_model(model, path):
    '''
    Save model using sklearn joblib.
    '''
    check_makedir(path)
    joblib.dump(model, path)


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