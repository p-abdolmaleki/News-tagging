import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import keras
from keras import layers
from keras.saving import load_model
import json



def preprocessing_1(new_df):
    stopwords = nltk.corpus.stopwords.words('english')
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    dataset = pd.DataFrame(columns=['text','label'])
    for index, row in new_df.iterrows():
        headline_description = row['Headline'] + ' ' + row['Description']
        headline_description_tokenized = nltk.word_tokenize(headline_description)
        headline_description_tokenized_filtered = [word.lower() for word in headline_description_tokenized if not word.lower() in stopwords and word.isalnum()]
        headline_description_tokenized_filtered_lammetized = [WordNetLemmatizer().lemmatize(word) for word in headline_description_tokenized_filtered]
        if len(headline_description_tokenized_filtered_lammetized) > 5:
            dataset.loc[index] = {
                'text' : ' '.join(headline_description_tokenized_filtered_lammetized),
                'label' : row['Label']
            }
    dataset['label'] = dataset['label'].map(class_indices)
    return dataset

  
vectorize_layer_model = load_model('models/vectorize_layer_model.keras')
vectorize_layer = vectorize_layer_model.layers[0]
def preprocessing_2(text):
    return(vectorize_layer(text))


class Transformer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.Attention = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.FeedForward = keras.models.Sequential([
            layers.Dense(units=self.ff_dim, activation='relu'),
            layers.Dense(units=self.embed_dim)
        ])
        self.Normalization_1 = layers.LayerNormalization(epsilon=1e-6)
        self.Normalization_2 = layers.LayerNormalization(epsilon=1e-6)
        self.Droupout_1 = layers.Dropout(self.rate)
        self.Droupout_2 = layers.Dropout(self.rate)
    def call(self, inputs):
        att_out = self.Attention(inputs, inputs)
        att_out = self.Droupout_1(att_out)
        out_1 = self.Normalization_1(inputs + att_out)
        out_ff = self.FeedForward(out_1)
        out_ff = self.Droupout_2(out_ff)
        out = self.Normalization_2(out_1 + out_ff)
        return out
    

class PositionalTokenEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tokenEmbeding = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.positionalEmbeding = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)
    def call(self, inputs):
        positions = np.arange(start=0, stop=self.maxlen, step=1)
        positions = self.positionalEmbeding(positions)
        tokens = self.tokenEmbeding(inputs)
        return positions + tokens