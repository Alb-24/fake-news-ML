from abc import ABC

import numpy
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Create directory where store the DataSets
os.makedirs('Dataset', exist_ok=True)
# Select GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Pytorch Dataset
class NewsDataSet(Dataset, ABC):
    def __init__(self, csv_file):
        """
        Arguments:
            csv_file (string): Path to the out.csv file
        """
        self.txt = pd.read_csv(filepath_or_buffer=csv_file, usecols=["content"], sep=",")
        self.labels = pd.read_csv(filepath_or_buffer=csv_file, usecols=["fake"], sep=",")
        self.max_words = self.get_max_words()

    def get_max_words(self):
        # return the max number of words in a column (i.e. in an article's content)
        length_of_the_content = self.txt["content"].str.split("\\s+")
        return length_of_the_content.str.len().max()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return a single item content and label
        label = self.labels[idx]
        text = self.txt[idx]
        item = {"Text": text, "Class": label}
        return item

    def to_csv(self):
        # TEST, UTILIZZATA PER VEDERE COSA SALVA IN TXT E LABELS
        self.txt.to_csv('DataSet/txt.csv')
        self.labels.to_csv('DataSet/labels.csv')


# Function to retur a token for the input (npy array, containing text)
def yield_tokens(np_array):
    """
    ravel() is needed to reshape the np_array from [[text1],[text2],[text3],...] to [text1, text2, text3,...],
    otherwise iteration is not possible
    """
    for article_text in np_array.ravel():
        yield tokenizer(article_text)


# If the out.csv, containing concateneted True and Fake news, is not created, then proceed to create it
if not os.path.exists("DataSet/out.csv"):
    # Load Datasets as dataframes (df), reading csv path file of true and fake datadets
    df_true = pd.read_csv("True.csv")
    df_fake = pd.read_csv("Fake.csv")

    # Add a column to store the "fake" label (0=true news, 1=fake news)
    df_true['fake'] = 0
    df_true.head()

    df_fake['fake'] = 1
    df_fake.head()

    # Concatenate both datasets into one unique
    df = pd.concat([df_fake, df_true]).reset_index(drop=True)

    # Combine title and text of the articles,
    # as it is easier to manipulate them together
    df['content'] = df['title'] + ' ' + df['text']

    # Join the content words as a string
    df['content'] = df['content'].apply(lambda x: "".join(x))

    # Remove the column of articles' date, because it's useless
    df.drop(columns=['date'], inplace=True)
    # Save out.csv file
    df.to_csv('DataSet/out.csv')

    print(df["content"][0])

# passing the dataframe to NewsDataSet
news_dataset = NewsDataSet("DataSet/out.csv", )

# Obtain the features and the labels
X = news_dataset.txt.values
Y = news_dataset.labels.values

# Random split in test and train data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

"""
# TESTING THE OUTPUTS
print("Train: ", x_train[0:2])
print("Label: ", y_train[0:2])
print("Test: ", x_test[0:2])
print("Label: ", y_test[0:2])
"""

# Initialize tokenizer with english language
tokenizer = get_tokenizer('basic_english')

vocab = build_vocab_from_iterator(yield_tokens(x_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# text pipeline will convert the article's content (str) to a list of tokens (list of int), based on the vocabolary
# lookup table
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
"""
example: print(text_pipeline("Donald Trump")) 
>> [75, 16]
"""
