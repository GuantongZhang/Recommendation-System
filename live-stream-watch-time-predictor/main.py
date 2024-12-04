from sklearn import linear_model
import numpy as np
import random
import pandas as pd
#from surprise import SVD, Reader, Dataset
#from surprise.model_selection import train_test_split
#from implicit import bpr
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

import csv

import gzip
from collections import defaultdict

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')

def CSV(path):
    f = open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def accuracy(predictions, y):
    correct = predictions == y
    Acc = sum(correct) / len(correct)
    return Acc

df = pd.read_csv("100k_a.csv",header=None)
df['interval'] = df[4] - df[3]

# We decide to mix our database to avoid over fitting because the database is
# in the user id order. And when we split our database to evaluate it, we won't
# have user id values that are different and large enough for the evaluation.

df_sample = df.sample(100000).reset_index(drop=True)
df_sample.columns = ['user', 'stream_id', 'streamer', 'start', 'stop', 'interval']

df_sample.to_csv('100k_a_sample.csv', index=False)

data = []
with open('100k_a_sample.csv', mode ='r') as file:
    file.readline()
    csvFile = csv.reader(file)
    for lines in csvFile:
        data.append(lines[0:3] + [float(lines[-1])])

# First, to evaluate my data, we will divide my data into a training set of the
# first 90,000 rows and a test set of the last 10,000. We will then predict
# with the trained model on the training data set on the test set. 

# To make a good prediction model, we had to create a data corresponding to the
# subtraction of the Start_time and Stop_time columns. This pre-processing of
# the data will allow us to give us a good information serving as a value to
# predict. This value corresponds to the time spent by me on the stream.

# We will create our model using the values of user_id, stream_id for the
# features and the value of the difference between Start_time and Stop_time for
# our label.

# To test it, we will evaluate its accuracy on the test set after training on
# the train set.

train = data[:80000]
valid = data[80000:90000]
test = data[90000:]
train_df = df_sample[:80000]
valid_df = df_sample[80000:90000]
test_df = df_sample[90000:]

ratings_per_user = defaultdict(list)
for d in train:
    ratings_per_user[d[0]].append(d[-1])
ratings_per_user = {u: np.mean(ratings_per_user[u]) for u in ratings_per_user}

ratings_per_stream = defaultdict(list)
for d in train:
    ratings_per_stream[d[1]].append(d[-1])
ratings_per_stream = {u: np.mean(ratings_per_stream[u]) for u in ratings_per_stream}

ratings_per_streamer = defaultdict(list)
for d in train:
    ratings_per_streamer[d[2]].append(d[-1])
ratings_per_streamer = {u: np.mean(ratings_per_streamer[u]) for u in ratings_per_streamer}

def feature(datum):
    global_mean = train_df['interval'].mean()
    
    if datum[0] in ratings_per_user:
        user_avg = ratings_per_user[datum[0]]
    else:
        user_avg = global_mean

    if datum[1] in ratings_per_stream:
        stream_avg = ratings_per_stream[datum[1]]
    else:
        stream_avg = global_mean   

    if datum[2] in ratings_per_streamer:
        streamer_avg = ratings_per_streamer[datum[2]]
    else:
        streamer_avg = global_mean       
    
    feat = [1, user_avg, stream_avg, streamer_avg]

    return feat


X_train = [feature(x) for x in train]
y_train = [x[-1] for x in train]

X_valid = [feature(x) for x in valid]
y_valid = [x[-1] for x in valid]

X_test = [feature(x) for x in test]
y_test = [x[-1] for x in test]

model_ = linear_model.LinearRegression()
model_.fit(X_train,y_train)

# Predict
pred_valid = model_.predict(X_valid)
MSE(y_valid, pred_valid)

userIDs = {}
itemIDs = {}
interactions = []

for d in data:
    u = d[0]
    i = d[1]
    r = d[2]
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
    interactions.append((u,i,r))

interactionsTrain  = interactions[:90000]
interactionsTest  = interactions[90000:]

itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,i,r in interactionsTrain:
    itemsPerUser[u].append(i)
    usersPerItem[i].append(u)

mu = sum([r for _,_,r in interactionsTrain]) / len(interactionsTrain)

optimizer = tf.keras.optimizers.Adam(0.1)

class LatentFactorModelBiasOnly(tf.keras.Model):
    def __init__(self, mu, lamb):
        super(LatentFactorModelBiasOnly, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i]
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) +\
                            tf.reduce_sum(self.betaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        pred = self.alpha + beta_u + beta_i
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)
    
modelBiasOnly = LatentFactorModelBiasOnly(mu, 0.00001)

def trainingStepBiasOnly(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,r = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)

        loss = model(sampleU,sampleI,sampleR)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
        (grad, var) in zip(gradients, model.trainable_variables)
        if grad is not None)
    return loss.numpy()

for i in range(50):
    obj = trainingStepBiasOnly(modelBiasOnly, interactionsTrain)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

pred = [modelBiasOnly.predict(userIDs[d[0]], itemIDs[d[1]]).numpy() for d in interactionsTest]
label = [d[2] for d in interactionsTest]
MSE(pred,label)