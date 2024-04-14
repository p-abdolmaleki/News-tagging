import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import accuracy_score
from keras.saving import load_model
from utils import preprocessing_1, preprocessing_2, Transformer, PositionalTokenEmbedding


datas = pd.read_csv('test.csv').sample(1000).reset_index(drop=True)
new_data = preprocessing_1(datas)
X = preprocessing_2(new_data['text'].values)
y_true = new_data['label'].values


# prams we need to define models
max_features = 20000
embedding_dim = 128
sequence_length = 80
num_classes = 4
num_heads = 8
units = 128
epochs = 5
metrics = ['accuracy',
           keras.metrics.F1Score(average='weighted')]

# define model 3
inputs = keras.Input(shape=(None,), dtype='int64', name='InputLayer')
x = layers.Embedding(max_features, embedding_dim, name='Embedding')(inputs)
x = layers.Dropout(0.5, name='Dropout1')(x)

x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, name='LSTM1'), name='Bidirectional1')(x)
x = layers.Bidirectional(layers.LSTM(units, name='LSTM2'), name='Bidirectional2')(x)

x = layers.Dense(units, activation='relu', name='Dense1')(x)
x = layers.Dropout(0.5, name='Dropout2')(x)

predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

model_3 = keras.Model(inputs, predictions)

# define model 5
inputs = keras.Input(shape=(None,), dtype='int64', name='InputLayer')

x = layers.Embedding(max_features, embedding_dim, name='Embedding')(inputs)
x = layers.Dropout(0.5, name='Dropout1')(x)

x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, name='Attention')(query=x, value=x)

x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, name='LSTM1'), name='Bidirectional1')(x)
x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, name='LSTM2'), name='Bidirectional2')(x)
x = layers.GlobalAveragePooling1D(name='GlobalAveragePooling1D')(x)

x = layers.Dense(units, activation='relu', name='Dense')(x)
x = layers.Dropout(0.5, name='Dropout2')(x)

predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

model_5 = keras.Model(inputs, predictions)

# define model 6
inputs = keras.Input(shape=(None,), dtype='int64', name='InputLayer')

embedding_layer = PositionalTokenEmbedding(sequence_length, max_features, embedding_dim)
x = embedding_layer(inputs)

transformer_layer_1 = Transformer(embedding_dim, num_heads, units)
x = transformer_layer_1(x)
transformer_layer_2 = Transformer(embedding_dim, num_heads, units)
x = transformer_layer_2(x)

x = layers.GlobalAveragePooling1D(name = 'GlobalAveragePooling1D')(x)

x = layers.Dense(units, activation="relu", name='Dense')(x)
x = layers.Dropout(0.5, name='Dropout1')(x)

predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

model_6 = keras.Model(inputs, predictions)

#loading models

# model_1 = load_model('models/model_1.keras')
# model_2 = load_model('models/model_2.keras')
# model_3.load_weights('models/model_3.weights.h5')
# model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
# model_4 = load_model('models/model_4.keras')
# model_5.load_weights('models/model_5.weights.h5')
# model_5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
model_6.load_weights('models/model_6.weights.h5')
model_6.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)


model = model_6
y_pred = np.argmax(model.predict(X), axis=1)
print('final model accuracy :{:.2f}'.format(accuracy_score(y_true, y_pred) * 100))

