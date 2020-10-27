import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.style as style
from keras.utils import to_categorical


all_cols = ['sepal_length','sepal_width','petal_length','petal_width','species']

ds = pd.read_csv('iris_ds.csv')
ds = ds.reindex(np.random.permutation(ds.index))
train = ds


s = StandardScaler()
for x in ds.columns:
    if x != 'species':
        train[x] = s.fit_transform(train[x].values.reshape(-1, 1)).astype('float64')


label_encoder = LabelEncoder()
train['species'] = label_encoder.fit_transform(train['species'])


train_features = train.drop('species',axis=1)
cat_label = train.pop('species').values
train_label = keras.utils.to_categorical(cat_label)


input_dim = train_features.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(8, input_dim = input_dim, activation=tf.keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(16,  activation=tf.keras.layers.LeakyReLU()))
model.add(keras.layers.Dense(3, activation='softmax'))


model.compile(optimizer='adam', loss='CategoricalCrossentropy',
              metrics = 'accuracy')


history = model.fit(train_features, train_label, epochs=200, validation_split=0.7)


results = model.evaluate(train_features, train_label)
print('\nLoss, Categorical Crossentropy Accuracy: \n',(results))


style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Categorical Crossentropy',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show()
 

