import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        Return two parameters including:
            model: the number of co-occurrence of each pair
            features: the pair of each pattern
        E.g.,
            model: (I -> am): 10, (I -> want): 5, ...
        '''
        # TO-DO 1-1: (You can use the hint code between begin and end or just delete it.)
        # begin your code


        count_bigram = defaultdict(int)

        bigrams = []

        total = 0
        for document in corpus_tokenize:
            for idx in range(len(document) - 1):
                total += 1
                bigram = (document[idx], document[idx + 1])
                bigrams.append(bigram)

                count_bigram[bigram] += 1


        model = defaultdict(lambda: defaultdict(int))


        for bigram in bigrams:
            x = bigram[0]
            y = bigram[1]
            model[x][y] += 1


        features = {k: v for k, v in sorted(count_bigram.items(), key=lambda item: -item[1])}

        self.count_bigram = count_bigram


        return model, features







        # end your code
        
        
        
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        perplexity = 0
        for document_tokenize in corpus:
            twograms = nltk.ngrams(document_tokenize, self.n)
            N = len(list(nltk.ngrams(document_tokenize, self.n)))
            probabilities = []
            for w1, w2 in twograms:
                numerator = 1 + self.model[w1][w2]
                denominator = sum(self.model[w1].values())
                # give a value to avoid divide-by-zero
                if denominator == 0:
                    probabilities.append(1e-3)
                else:
                    probabilities.append(numerator / denominator)

            cross_entropy = -1 / N * sum([math.log(p, 2) for p in probabilities])
            perplexity += math.pow(2, cross_entropy)
        
        perplexity /= len(corpus)
        return perplexity
    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # TO-DO 1-2
        # begin your code

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 500

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        self.train(df_train)

        # Record
        gram_index = {}
        gram_position = {}
        gram_neg = {}
        pos_number = 0
        neg_number = 0

        # Tokenize train corpus
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        y = list(df_train['sentiment'])

        # Select k_best features for chi-square feature selection
        k_best = list(self.features.items())[:feature_num]

        # Chi-square feature selection
        for i, document in tqdm(enumerate(train_corpus)):
            for idx in range(len(document) - 1):
                bigram = (document[idx], document[idx + 1])
                if y[i] == 1:
                    if bigram in gram_position:
                        gram_position[bigram] += 1
                    else:
                        gram_position[bigram] = 1
                    pos_number += 1
                else:
                    if bigram in gram_neg:
                        gram_neg[bigram] += 1
                    else:
                        gram_neg[bigram] = 1
                    neg_number += 1

        # Compute chi-squared statistics for each bigram
        All_number = pos_number + neg_number
        allBigram = {**gram_position, **gram_neg}
        chiFeatures = []

        for key in tqdm(allBigram):
            key_count = 0
            if key in gram_position:
                key_count += gram_position[key]
            if key in gram_neg:
                key_count += gram_neg[key]
            E_11 = pos_number * key_count / All_number
            E_10 = neg_number * key_count / All_number
            E_01 = pos_number * (All_number - key_count) / All_number
            E_00 = neg_number * (All_number - key_count) / All_number
            chi = ((gram_position[key] if key in gram_position else 0) - E_11) ** 2 / E_11
            chi += ((gram_neg[key] if key in gram_neg else 0) - E_10) ** 2 / E_10
            chi += (pos_number - (gram_position[key] if key in gram_position else 0) - E_01) ** 2 / E_01
            chi += (neg_number - (gram_neg[key] if key in gram_neg else 0) - E_00) ** 2 / E_00

            chiFeatures.append((key, chi))

        # Sort the chi-squared statistics in descending order and select the top k features
        chiFeatures = sorted(chiFeatures, key=lambda pair: -pair[1])
        k_best = chiFeatures[:feature_num]

        print(k_best[:20])

        # Tokenize test corpus
        test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]

        # Compute embeddings for train corpus and test corpus using k_best features
        train_corpus_embedding = [[0] * len(k_best) for _ in range(len(df_train['review']))]
        test_corpus_embedding = [[0] * len(k_best) for _ in range(len(df_test['review']))]

        for i, document in tqdm(enumerate(train_corpus)):
            for idx in range(len(document) - 1):
                bigram = (document[idx], document[idx + 1])
                if bigram in gram_index:
                    train_corpus_embedding[i][gram_index[bigram]] += 1

        for i, document in enumerate(test_corpus):
            for idx in range(len(document) - 1):
                bigram = (document[idx], document[idx + 1])
                if bigram in gram_index:
                    test_corpus_embedding[i][gram_index[bigram]] += 1


        '''
        train_tokens = [tokenizer.tokenize(sentence) for sentence in df_train['review']]
        test_tokens = [tokenizer.tokenize(sentence) for sentence in df_test['review']]

        train_embeddings = [[get_embedding(token) for token in tokens] for tokens in train_tokens]
        test_embeddings = [[get_embedding(token) for token in tokens] for tokens in test_tokens]

       

        train_features = Counter()
        for tokens in train_tokens:
            features = get_ngram_features(tokens)
            train_features.update(features)

        most_common_features = [x[0] for x in train_features.most_common(feature_num)]

     

        train_feature_vectors = []
        for tokens in train_tokens:
            features = get_features(tokens)
            feature_vector = [features[feature] for feature in most_common_features]
            train_feature_vectors.append(feature_vector)

        test_feature_vectors = []
        for tokens in test_tokens:
            features = get_features(tokens)
            feature_vector = [features[feature] for feature in most_common_features]
            test_feature_vectors.append(feature_vector)

        train_corpus_embedding = np.array(train_feature_vectors)
        test_corpus_embedding = np.array(test_feature_vectors)
        '''


        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
