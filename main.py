import os
import pandas as pd
import matplotlib.pyplot as plt

# %%

path = r"F:\Загрузки\sms-spam-corpus.csv"
df = pd.read_csv(path, encoding='latin1', names=["v1", "v2", "A", "B", "C"])
df.head(10)

# %%

import re

df['v2'].replace(regex=True, inplace=True, to_replace=r'[^a-zA-Z^ \t\n\r]', value=r'')
df['v2'] = df['v2'].astype(str).map(lambda x: x.lower())

df.head()['v2']

# %%

lineofsw = "''a,to,the,in,have,has,had,do,does,did,am,is,are,shall,will,should,would,may,might,must,can,could,a,to,the,in,be,being,been"

stop_words = lineofsw.split(',')

df['v2'] = df['v2'].apply(lambda x: ' '.join([word for word in x.split(' ') if word not in stop_words]))

df.head()['v2']

# %%

import nltk

ps = nltk.stem.SnowballStemmer('english')
df['v2'] = df['v2'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split(' ')]))

df.to_csv('dataset.csv', index=False)
df.head()['v2']

# %%

from collections import defaultdict

filewithham = open('ham.txt', 'w')
filewithspam = open('spam.txt', 'w')

ham = df[df['v1'] == 'ham']['v2']
spam = df[df['v1'] == 'spam']['v2']

hamdict = defaultdict(int)
spamdict = defaultdict(int)

for sentence in ham:
    for word in sentence.split():
        hamdict[word] += 1

for key, item in hamdict.items():
    filewithham.write('{} {}'.format(key, item) + '\n')

for sentence in spam:
    for word in sentence.split():
        spamdict[word] += 1

for key, item in spamdict.items():
    filewithspam.write('{} {}'.format(key, item) + '\n')

ham_len = hamdict.keys()
spam_len = spamdict.keys()

# %%

import seaborn as sns

fig, ax = plt.subplots()

ham_data = [len(x) for x in ham_len]  # data
filewithham = pd.DataFrame({'ham_len': ham_data})  # dataframe
fig = sns.distplot(filewithham['ham_len']);  # name
fig.figure.savefig('fig_1.png')  # save

# %%

plt.hist(filewithham['ham_len'])
plt.xlabel('ham words')
plt.ylabel('len')
plt.tight_layout()
plt.savefig('fig_2.png')

# %%

spam_data = [len(x) for x in spam_len]
filewithspam = pd.DataFrame({'spam_len': spam_data})

fig = sns.distplot(filewithspam['spam_len']);
fig.figure.savefig('fig_3.png')

# %%

plt.hist(filewithspam['spam_len'])
plt.xlabel('spam words')
plt.ylabel('len')
plt.tight_layout()
plt.savefig('fig_4.png')

# %%

import numpy as np

print('Ham mean: {} | Spam mean: {}'.format(np.mean(ham_data), np.mean(spam_data)))

# %%

ham_msg = [len(x) for x in ham]
filewithham = pd.DataFrame({'ham_msg': ham_msg})
fig = sns.distplot(filewithham['ham_msg']);
fig.figure.savefig('fig_5.png')

# %%

plt.hist(filewithham / filewithham.count())
plt.xlabel('ham msg')
plt.ylabel('len')
plt.tight_layout()
plt.savefig('fig_6.png')

# %%

spam_msg = [len(x) for x in spam]
filewithspam = pd.DataFrame({'spam_msg': spam_msg})
fig = sns.distplot(filewithspam['spam_msg']);
fig.figure.savefig('fig_7.png')

# %%

plt.hist(filewithspam['spam_msg'])
plt.xlabel('spam msg')
plt.ylabel('len')
plt.tight_layout()
plt.savefig('fig_8.png')

# %%

print('Ham msg mean: {} | Spam msg mean: {}'.format(np.mean(ham_msg), np.mean(spam_msg)))

# %%

list_ham = list(hamdict.items())
list_ham.sort(key=lambda i: i[1])
list_ham = list_ham[-20:]
list_ham = dict(list_ham)

list_spam = list(spamdict.items())
list_spam.sort(key=lambda i: i[1])
list_spam = list_spam[-20:]
list_spam = dict(list_spam)

# %%

plt.bar(list_ham.keys(), list_ham.values(), color='black')
plt.xticks(rotation=90);
plt.savefig('fig_9.png')

# %%

plt.bar(list_spam.keys(), list_spam.values(), color='blue')
plt.xticks(rotation=90);
plt.savefig('fig_10.png')

# %%


# %%


# %%


# %%


# %%


# %%


