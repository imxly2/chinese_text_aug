import numpy as np
from tqdm import tqdm
import jieba
import pickle
import os

stopwords = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop.txt'),
                 encoding='utf8').read().split('\n')
syn_words = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syn.pkl'), 'rb'))
max_try = 10


def synonym_candidate(word):
    # syn_words = syn_finder(word)
    # syn_words = [syn_words[0][i] for i, score_ in enumerate(syn_words[1]) if score_ > score and i != 0]
    return syn_words.get(word, [])


def word_prob(sent_cut):
    probs = []
    for i in sent_cut:
        if i in stopwords:
            probs.append(0)
        else:
            probs.append(1)
    probs = np.array(probs) / sum(probs)
    return probs


def synonym_replacement(sent_cut, probs, n=1):
    finished = 0
    sent_copy = sent_cut[:]
    tried = 0
    while finished < n and tried < max_try:
        tried += 1
        indices = np.random.choice(range(len(sent_cut)), p=probs)
        word = sent_cut[indices]
        syn_words = synonym_candidate(word)
        if syn_words:
            sent_copy[indices] = np.random.choice(syn_words)
            finished += 1
    return ''.join(sent_copy)


def random_insertion(sent_cut, probs, n=1):
    insertions = {}
    tried = 0

    while len(insertions) < n and tried < max_try:
        tried += 1
        indices1 = np.random.choice(range(len(sent_cut)), p=probs)
        word = sent_cut[indices1]
        syn_words = synonym_candidate(word)
        if syn_words:
            syn = np.random.choice(syn_words)
            indices2 = np.random.choice(range(len(sent_cut)))
            insertions[syn] = indices2
    sent_copy = sent_cut[:]
    for i, (syn, indices) in enumerate(sorted(insertions.items())):
        sent_copy.insert(indices + i, syn)
    return ''.join(sent_copy)


def random_swap(sent_cut, n=1):
    for i in range(n):
        indices1, indices2 = np.random.choice(range(len(sent_cut)), 2)
        sent_copy = sent_cut[:]
        sent_copy[indices1], sent_copy[indices2] = sent_copy[indices2], sent_copy[indices1]
    return ''.join(sent_copy)


def random_deletion(sent_cut, p=0.1):
    sent_copy = []
    for i in range(len(sent_cut)):
        if np.random.uniform() > p:
            sent_copy.append(sent_cut[i])
    return ''.join(sent_copy)


def _eda(sent, p=0.1):
    res = set()
    sent_cut = list(jieba.cut(sent, use_paddle=True))
    length = len(sent_cut)
    n = int(length * p)
    if n == 0:
        return []

    prob = word_prob(sent_cut)
    res.add(synonym_replacement(sent_cut, prob, n))
    res.add(random_insertion(sent_cut, prob, n))
    res.add(random_swap(sent_cut, n))
    res.add(random_deletion(sent_cut, p))
    return list(res)


def eda(inp, p=0.1):
    if isinstance(inp, str):
        return _eda(inp, p)
    else:
        res = []
        for sent in tqdm(inp):
            res.extend(_eda(sent, p))
    return res


if __name__ == '__main__':
    for i in eda('原作者虽给出了针对英文语料数据增强的代码实现，但不适合中文语料。我经过对原论文附上的代码的修改，现在推出这个适合中文语料的数据增强EDA的实现'):
        print(i)
