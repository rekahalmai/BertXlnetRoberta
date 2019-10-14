import tarfile
import wget
import os
from glob import glob
import numpy as np
import pandas as pd


def load_texts_labels_from_folders(path, folders):
    texts, labels = [], []
    for idx, label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(idx)
    return texts, np.array(labels).astype(np.int8)


def main():
    """
    Downloads and saves the imdb data in csv format.
    It also seperates the train, evaluation and test data and save it in the same folder.
    """

    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    imdb = wget.download(url)
    print('Download is done.')
    fname = 'aclImdb_v1.tar.gz'
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()

    try:
        os.rename('aclImdb', 'data')
        PATH = 'data/'
    except OSError:
        print('Renaming the downloaded folder to data has failed.')
        PATH = 'aclImdb'
    print(f'Data directory: {PATH}')

    print(f'Creating imdb...')
    names = ['neg', 'pos']
    # Train dataset
    trn, trn_y = load_texts_labels_from_folders(f'{PATH}train', names)
    d_train = {'text': trn, 'subset': 'train', 'label': trn_y, 'id': np.arange(25000)}
    df_train = pd.DataFrame(data=d_train)
    # Evaluation dataset
    val, val_y = load_texts_labels_from_folders(f'{PATH}test', names)
    d_val = {'text': val, 'subset': 'val', 'label': val_y, 'id': np.arange(25000, 50000)}
    df_val = pd.DataFrame(data=d_val)

    names = ['unsup']
    unsup, unsup_y = load_texts_labels_from_folders(f'{PATH}train', names)
    d_unsup = {'text': unsup, 'subset': 'unsup', 'label': unsup_y, 'id': np.arange(50000, 100000)}
    df_unsup = pd.DataFrame(data=d_unsup)

    imdb = pd.concat([df_train, df_val, df_unsup])
    imdb.to_csv('data/imdb.csv', sep='|', index=None)
    imdb['alpha'] = 'a'  # Not sure why add alpha (a string column) but I found it in the implementations.
    print(f'Created and saved imdb.csv...')

    print(f'Creating train, evaluation and test sets...')
    train = imdb[imdb.subset == 'train'].drop(columns=['subset'])
    train = train[['id', 'label', 'alpha', 'text']]
    evaluation = imdb[imdb.subset == 'val'].drop(columns=['subset'])
    evaluation = evaluation[['id', 'label', 'alpha', 'text']]
    test = imdb[imdb.subset == 'unsup'].drop(columns=['subset'])
    test = test[['id', 'alpha', 'text']]

    print(f'Saving train, evaluation and test sets to {PATH}...')
    train.to_csv(f"{PATH}train.tsv", sep="\t", index=False, header=True)
    evaluation.to_csv(f"{PATH}evaluation.tsv", sep="\t", index=False, header=True)
    test.to_csv(f"{PATH}test.tsv", sep="\t", index=False, header=True)
    print(f"Saved train, evaluation and test sets to {PATH}.")


if __name__ == "__main__":
    main()