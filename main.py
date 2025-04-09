import glob
import math
import os
import re
from collections import defaultdict

def load_data(directory: str):
    x = []
    y = []
    for f in glob.glob(os.path.join(directory, "HAM.*.txt")):
        with open(f, 'r') as file:
            x.append(file.read())
            y.append(0)

    for f in glob.glob(os.path.join(directory, "SPAM.*.txt")):
        with open(f, 'r') as file:
            x.append(file.read())
            y.append(1)
    return x, y

def _tokenize(text: str):
    text = text.lower()
    # remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)

    words = re.findall(r'\b[a-z]+\b', text)
    return words

def nb_train(x: list[str], y: list[int]):
    ham_count = 0
    spam_count = 0
    ham_fd = defaultdict(int)
    spam_fd = defaultdict(int)
    for doc, label in zip(x, y):
        words = _tokenize(doc)
        if label == 1:
            spam_count += 1
            for word in words:
                spam_fd[word] += 1
        else:
            ham_count += 1
            for word in words:
                ham_fd[word] += 1

    return {
        'ham_count': ham_count,
        'spam_count': spam_count,
        'ham_fd': ham_fd,
        'spam_fd': spam_fd,
    }

def _classify(doc: str, get_probs, use_log: bool, vocabulary: set[str], prob_spam: float, prob_ham: float):
    if use_log:
        prob_spam = math.log(prob_spam)
        prob_ham = math.log(prob_ham)

    for word in _tokenize(doc):
        if word not in vocabulary:
            continue
        prob_spam_i, prob_ham_i = get_probs(word)
        if use_log:
            prob_spam += math.log(prob_spam_i) if prob_spam_i != 0 else 0
            prob_ham += math.log(prob_ham_i) if prob_ham_i != 0 else 0
        else:
            prob_spam *= prob_spam_i
            prob_ham *= prob_ham_i

    if prob_spam > prob_ham:
        return 1
    return 0

def nb_test(docs: list[str], trained_model, use_log=False, smoothing=False):
    prob_spam = trained_model['spam_count'] / len(docs)
    prob_ham = trained_model['ham_count'] / len(docs)
    vocabulary = trained_model['ham_fd'].keys() | trained_model['spam_fd'].keys()
    print(vocabulary)
    vocabulary_size = len(vocabulary)
    spam_total_word_count = sum(v for v in trained_model['spam_fd'].values())
    ham_total_word_count =  sum(v for v in trained_model['ham_fd'].values())

    def get_word_probs(word: str):
        spam_count = trained_model['spam_fd'].get(word, 0)
        ham_count = trained_model['ham_fd'].get(word, 0)
        if smoothing:
            spam_prob = (spam_count + 1) / (spam_total_word_count + vocabulary_size)
            ham_prob = (ham_count + 1) / (ham_total_word_count + vocabulary_size)
        else:
            spam_prob = spam_count / spam_total_word_count
            ham_prob = ham_count / ham_total_word_count
        return spam_prob, ham_prob

    results = []
    for doc in docs:
        label = _classify(doc, get_word_probs, use_log, vocabulary, prob_spam, prob_ham)
        results.append(label)
    return results

def f_score(y_true: list[int], y_pred: list[int]):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    # f1 = 2tp / (2tp + fp + fn)
    return (2 * tp) / (2 * tp + fp + fn)

def main():
    x_train, y_train = load_data("./SPAM_training_set/")

    model = nb_train(x_train, y_train)


    x_test, y_test = load_data("./SPAM_test_set/")

    y_pred = nb_test(x_test, model, use_log=False, smoothing=False)
    print(f'Log=False, Smoothing=False, F1 Score={f_score(y_test, y_pred)}')

    y_pred = nb_test(x_test, model, use_log=False, smoothing=True)
    print(f'Log=False, Smoothing=True, F1 Score={f_score(y_test, y_pred)}')

    y_pred = nb_test(x_test, model, use_log=True, smoothing=False)
    print(f'Log=True, Smoothing=False, F1 Score={f_score(y_test, y_pred)}')

    y_pred = nb_test(x_test, model, use_log=True, smoothing=True)
    print(f'Log=True, Smoothing=True, F1 Score={f_score(y_test, y_pred)}')


if __name__ == '__main__':
    main()