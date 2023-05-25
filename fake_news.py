import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import nltk


def preprocess_text(text: str):
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
    Returns:
        str: the cleaned list of tokens
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    # 1. tokenize
    tokens = nltk.word_tokenize(text)
    # 2. check if stopword
    tokens = [w.lower() for w in tokens if not w in stopwords.words("english")]
    return tokens


# Load Datasets
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

# Add a column to store the "fake" label (0=true news, 1=fake news)
df_true['fake'] = 0
df_true.head()

df_fake['fake'] = 1
df_fake.head()

# Concatenate both datasets into one unique
df = pd.concat([df_true, df_fake]).reset_index(drop=True)

# Combine title and text of the articles,
# as it is easier to manipulate them together
df['content'] = df['title'] + ' ' + df['text']

# Remove the column of articles' date, because it's useless
df.drop(columns=['date'], inplace=True)

# Splitting Data Into Test And Train
x_train, x_test, y_train, y_test = train_test_split(df.content, df.fake, test_size=0.2, shuffle=True)

# init the tokenizer with an out_of_vocabulary token
tokenizer = Tokenizer(oov_token="<OOV>")

# generate word indexes
tokenizer.fit_on_texts(x_train)

# generate sequences and apply padding
sequences = tokenizer.texts_to_sequences(df.content)
padded = pad_sequences(sequences, padding='post')

print("The encoding for document\n", df.content[0], "\n is : \n\n", padded[0])
