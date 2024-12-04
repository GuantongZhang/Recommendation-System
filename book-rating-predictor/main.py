# starter
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model

import warnings
warnings.filterwarnings("ignore")

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r
        
punctuation = set(string.punctuation)

# preparing
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
global_mean = np.mean([r for u, b, r in allRatings])


ratingsTrain = allRatings[:197500]
ratingsValid = allRatings[197500:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# create negative training
ratingsTrain_neg = [(u, get_random_book(u), 0) for u, b, r in ratingsTrain]
ratingsTrain_pos = [(u, b, 1) for u, b, r in ratingsTrain]
ratingsTrain_all = ratingsTrain_pos + ratingsTrain_neg

# create negative validation
def get_random_book(user):
    while True:
        book = np.random.choice(list(ratingsPerItem.keys()))
        if not book in [b for b, r in ratingsPerUser[user]]:
            break # worst case loop forever
    return book

ratingsValid_neg = [(u, get_random_book(u), 0) for u, b, r in ratingsValid]
ratingsValid_pos = [(u, b, 1) for u, b, r in ratingsValid]
ratingsValid_all = ratingsValid_pos + ratingsValid_neg


# predict_read functions
def Jaccard(s1, s2):
    return len(np.intersect1d(list(s1), list(s2))) / len(np.union1d(list(s1), list(s2)))

def feature_read(tup):
    user = tup[0]
    item = tup[1]
    mean_on_user = np.mean([x[1] for x in ratingsPerUser[user]])
    mean_on_item = np.mean([x[1] for x in ratingsPerItem[item]])
    if np.isnan(mean_on_user):
        mean_on_user = mean_on_item
    if np.isnan(mean_on_item):
        mean_on_user = global_mean
        mean_on_item = global_mean

    return [
        mean_on_user,
        len([x[1] for x in ratingsPerUser[user]]),
        mean_on_item,
        len([x[1] for x in ratingsPerItem[item]]),
        1
    ]

def pred_read(user, item):
    tup = (user, item)
    pred = clf_read.predict(np.array(feature_read(tup)).reshape(1, -1))
    return pred[0]

# validation
X_train_read = [feature_read(d) for d in ratingsTrain_all]
y_train_read = [d[2] for d in ratingsTrain_all]
X_valid_read = [feature_read(d) for d in ratingsValid_all]
y_valid_read = [d[2] for d in ratingsValid_all]

c = 1
clf_read = linear_model.LogisticRegression(C=1, class_weight={0:1, 1:1.78}).fit(X_train_read, y_train_read)
preds = clf_read.predict(X_valid_read)
acc = (np.array(y_valid_read) == np.array(preds)).mean()


# prediction
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    predictions.write(u + ',' + b + ',' + str(pred_read(u, b)) + '\n')
predictions.close()


data = []
for d in readGz("train_Category.json.gz"):
    data.append(d)
reviews_train = data[:90000]
reviews_valid = data[90000:]

word_count = defaultdict(int)
total_word = 0
for d in reviews_train:
    r = d['review_text'].lower().translate(str.maketrans('', '', string.punctuation)).split()
    for w in r:
        total_word += 1
        word_count[w] += 1
counts = [(c, w) for w, c in sorted(word_count.items(), key=lambda item: item[1], reverse=True)]

words_num = 1000
words = [x[1] for x in counts[:words_num]]
word_id = dict(zip(words, range(len(words))))
word_set = set(words)

# bag-of-word for all genres
bags_per_genre = [[], [], [], [], []]
for d in data:
    words = d['review_text'].lower().translate(str.maketrans('', '', string.punctuation)).split()
    if d['genre'] == 'children':
        bags_per_genre[0] += words
    elif d['genre'] == 'comics_graphic':
        bags_per_genre[1] += words
    elif d['genre'] == 'fantasy_paranormal':
        bags_per_genre[2] += words
    elif d['genre'] == 'mystery_thriller_crime':
        bags_per_genre[3] += words
    else:
        bags_per_genre[4] += words

word_count_0 = defaultdict(int)
for w in bags_per_genre[0]:
    word_count_0[w] += 1
counts_0 = [(c, w) for w, c in sorted(word_count_0.items(), key=lambda item: item[1], reverse=True)]

word_count_lst = []
counts_lst = [0] * 5
for i in range(5):
    word_count_lst.append(defaultdict(int))
    for w in bags_per_genre[i]:
        word_count_lst[i][w] += 1
    counts_lst[i] = [(c, w) for w, c in sorted(word_count_lst[i].items(), key=lambda item: item[1], reverse=True)]

top_words_0 = [x[1] for x in counts_lst[0][:100]]
top_words_1 = [x[1] for x in counts_lst[1][:100]]
top_words_2 = [x[1] for x in counts_lst[2][:100]]
top_words_3 = [x[1] for x in counts_lst[3][:100]]
top_words_4 = [x[1] for x in counts_lst[4][:100]]

def feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    ws = r.split()
    ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
    ws3 = [' '.join(x) for x in list(zip(ws[:-2],ws[1:-1],ws[2:]))]
    for w in ws + ws2 + ws3:
        if w in wordSet:
            feat[wordId[w]] += 1
    feat += [1,
             Jaccard(ws, top_words_0),
             Jaccard(ws, top_words_1),
             Jaccard(ws, top_words_2),
             Jaccard(ws, top_words_3),             
             Jaccard(ws, top_words_4)
            ]
    
    return feat

# training
words = [x[1] for x in counts[:7000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

X_train = [feature(d) for d in reviews_train]
y_train = [d['genre'] for d in reviews_train]

X_valid = [feature(d) for d in reviews_valid]
y_valid = [d['genre'] for d in reviews_valid]


# validation
clf = linear_model.LogisticRegression(C=7).fit(X_train, y_train)
acc = (np.array(y_valid) == np.array(clf.predict(X_valid))).mean()

# prediction
catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}

predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")

for l in readGz("test_Category.json.gz"):
    cat = catDict[clf.predict(np.array(feature(l)).reshape(1, -1))[0]]
    predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(cat) + "\n")
predictions.close()
