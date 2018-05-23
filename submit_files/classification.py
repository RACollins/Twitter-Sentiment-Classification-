#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import csv
import re
import nltk
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


#regular expressions and tokeniser and lemmatizer
lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case = True)
regExURL = re.compile(r"http[s]?\S+")
regExUserMentions = re.compile(r"@\S+")
regExAmp = re.compile(r"&amp;")
regExNonAlpha = re.compile(r"([^\s\w]|_)+")
regExEmoji = re.compile("[\U0001F600-\U0001F64F]|\
                            [\U00002702-\U000027B0]|\
                            [\U0001F680-\U0001F6C0]|\
                            [\U0001F300-\U0001F5FF]|\
                            [\U000024C2-\U0001F251]|\
                            [\U0001F1E0-\U0001F1FF]", flags = re.UNICODE)
regExElongated = re.compile(r"\w*(?=(\w)\1{2,})\w*")
regExLaughText = re.compile(r"\b(a*ha+h[ha]*|l+o+l+[ol]*|e*he+h[he]*)\b",\
                            flags = re.IGNORECASE)
regExAllCaps = re.compile(r"\b[A-Z]{4,}\b")
regExQuestionExclaim = re.compile(r"(\!{2,}|\?{2,}|\?\!+|\!\?+)1*")
regExEllipsis = re.compile(r"\.\.\.\.*")
regExPosSmile = re.compile(r":-\)|: \)|:\)|:D|:p|:o\)|:]|:3|:c\)|:>|=]|8\)|;\)|; \)|\^\-\^|\^_\^|\^\.\^")
regExNegSmile = re.compile(r">:\[|:-\(|:\(|: \(|:-c|:c|:-<|:<|:-\[|:\[|:{|:’-\(|:’\(|:'-\(|:'\(|>\.<|>_<|>\-<")
regExHashtag = re.compile(r"#\w+")

#define functions to process tweets
def replace_url(doc):
    processed_doc = regExURL.sub(" urllink ", doc)
    return(processed_doc)

def replace_user_mention(doc):
    processed_doc = regExUserMentions.sub(" usermention ", doc)
    return(processed_doc)

def replace_amp(doc):
    processed_doc = regExAmp.sub("&", doc)
    return(processed_doc)

def replace_hashtag(doc):
    processed_doc = regExHashtag.sub(" hashtag ", doc)
    return(processed_doc)

def remove_non_alpha(doc):
    processed_doc = regExNonAlpha.sub("", doc)
    return(processed_doc)

def replace_emoji(doc):
    processed_doc = regExEmoji.sub(" emoji ", doc)
    return(processed_doc)

def replace_elongated_words(doc):
    processed_doc = regExElongated.sub(" elongatedword ", doc)
    return(processed_doc)

def replace_laugh_text(doc):
    processed_doc = regExLaughText.sub(" laughtext ", doc)
    return(processed_doc)

def replace_all_caps(doc):
    processed_doc = regExAllCaps.sub(" allcaps ", doc)
    return(processed_doc)

def replace_question_exclaim(doc):
    processed_doc = regExQuestionExclaim.sub(" questionexclaim ",doc)
    return(processed_doc)

def replace_ellipsis(doc):
    processed_doc = regExEllipsis.sub(" ellipsis ", doc)
    return(processed_doc)

def replace_pos_smile(doc):
    processed_doc = regExPosSmile.sub(" positiveface ", doc)
    return(processed_doc)

def replace_neg_smile(doc):
    processed_doc = regExNegSmile.sub(" negativeface ", doc)
    return(processed_doc)

def tokenise(doc):
    tokenised_doc = tknzr.tokenize(doc)
    return(tokenised_doc)

#get the term frequency-inverse-document-frequency of the tweets
vectorizer = TfidfVectorizer(stop_words = "english")
def getUnigrams(vec, training_corpus = None, testing_corpus = None):
    if training_corpus:
        X = vec.fit_transform(training_corpus)
        freq_array = X.todense()
        vocab_dict = vectorizer.vocabulary_
    elif testing_corpus:
        X = vec.transform(testing_corpus)
        freq_array = X.todense()
        vocab_dict = vectorizer.vocabulary_
    return(vocab_dict, freq_array)

#import pos/neg word lists and make combined dictionary
with open("negative-words.txt", encoding = 'ISO-8859-1') as f_neg, open("positive-words.txt") as f_pos:
    negativeTxtDoc, positiveTxtDoc = f_neg.read(), f_pos.read()

#remove description in head of text file
pos_words, neg_words = positiveTxtDoc[1537:], negativeTxtDoc[1541:]

#apply regular expressions to match word format of corpus
pos_words, neg_words = remove_non_alpha(pos_words), remove_non_alpha(neg_words) #remove non-alphanumeric characters
pos_words, neg_words = tokenise(pos_words), tokenise(neg_words)

#combined dictionary of negative and positive words
pos_dict, neg_dict = {k:"positiveword" for k in pos_words}, {k:"negativeword" for k in neg_words}
pos_neg_dict = pos_dict.copy()
pos_neg_dict.update(neg_dict)

def replace_pos_neg_words(doc):
    processed_doc = [pos_neg_dict[token] if token in pos_neg_dict else token for token in doc]
    return(processed_doc)

#open twitter lexicon text file and save as a dictionary
twitterLexiconDict = {}
with open("twitter-lexicon.txt", "r") as tlf:
    for line in tlf:
        entry = line.strip().split("\t")
        twitterLexiconDict[entry[0]] = float(entry[1])

#function to process tweets and create a feature space
def get_feature_space(tweetList, training = True):
    processed_tweets, y, idVec, lexicon_vector, count_vector = [], [], [], [], []
    for tweet in tweetList:

        #get label and tweet id vectors
        y.append(tweet[1])
        idVec.append(tweet[0])

        #use twitter lexicon to score each tweet, return a vector of all scores
        tokenised_raw_tweet = tokenise(tweet[2])
        tweetScore = []
        wordScore = 0
        for word in tokenised_raw_tweet:
            if word in twitterLexiconDict:
                wordScore += twitterLexiconDict[word]
        tweetScore.append(wordScore)

        #replace using functions above
        processed_tweet = replace_user_mention(tweet[2])
        processed_tweet = replace_url(processed_tweet)
        processed_tweet = replace_amp(processed_tweet)
        processed_tweet = replace_pos_smile(processed_tweet)
        processed_tweet = replace_neg_smile(processed_tweet)
        processed_tweet = replace_hashtag(processed_tweet)
        processed_tweet = replace_emoji(processed_tweet)
        processed_tweet = replace_ellipsis(processed_tweet)
        processed_tweet = replace_question_exclaim(processed_tweet)

        #now remove remaining punctuation and continue
        processed_tweet = remove_non_alpha(processed_tweet)
        processed_tweet = replace_laugh_text(processed_tweet)
        processed_tweet = replace_elongated_words(processed_tweet)
        processed_tweet = replace_all_caps(processed_tweet)

        #now lower case and continue
        processed_tweet = processed_tweet.lower()
        tokenized_tweet = tokenise(processed_tweet)
        tokenized_tweet = replace_pos_neg_words(tokenized_tweet)
        lemmatised_tweet = [lemmatizer.lemmatize(token) for token in tokenized_tweet]
        processed_tweets.append(" ".join(lemmatised_tweet))
        lexicon_vector.append(tweetScore)
        count_vector.append([float(len(tokenized_tweet))])
    if training == True:
        unigram_positions, unigram_array = getUnigrams(vectorizer, training_corpus = processed_tweets)
    elif training == False:
        unigram_positions, unigram_array = getUnigrams(vectorizer, testing_corpus = processed_tweets)

    return(idVec, unigram_positions, unigram_array, np.array(lexicon_vector), np.array(count_vector), y)

#import traing data to a list of lists
trainingTweets = []
with open("twitter-training-data.txt", "r") as f:
    for line in f:
        tweet = line.strip().split("\t")
        trainingTweets.append(tweet)

#creating feature space to be used in fitting
print("Creating feature space, please wait...")
trainingIDs, mapping_train, X_train, lexicon_scores_train, token_count_train, y_train = get_feature_space(trainingTweets)

for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']:
    if classifier == 'myclassifier1':
        print('Training ' + classifier)

        #define features to be used in the classifier and reduce feature space
        classifier_features = ["urllink", "usermention", "hashtag", "elongatedword", "emoji", "laughtext", "allcaps", "ellipsis", "questionexclaim", "positiveword", "negativeword", "positiveface", "negativeface"]
        feature_columns_trainset = [mapping_train[feature] for feature in classifier_features]
        reduced_X_train = X_train[:, feature_columns_trainset]

        #add columns for lexicon scores and token counts
        reduced_X_train = np.append(reduced_X_train, lexicon_scores_train, axis = 1)
        reduced_X_train = np.append(reduced_X_train, token_count_train, axis = 1)

        #generate random forest classifier and fit training data
        rfc = RandomForestClassifier()
        rfc.fit(reduced_X_train, y_train)
        print("Training done!")

    elif classifier == 'myclassifier2':
        print('Training ' + classifier)

        #define features to be used in the classifier and reduce feature space
        classifier_features = ["urllink", "usermention", "hashtag", "elongatedword", "emoji", "laughtext", "allcaps", "ellipsis", "questionexclaim", "positiveword", "negativeword", "positiveface", "negativeface"]
        feature_columns_trainset = [mapping_train[feature] for feature in classifier_features]
        reduced_X_train = X_train[:, feature_columns_trainset]

        #add columns for lexicon scores and token counts
        reduced_X_train = np.append(reduced_X_train, lexicon_scores_train, axis = 1)
        reduced_X_train = np.append(reduced_X_train, token_count_train, axis = 1)

        #generate random forest classifier and fit training data
        rfc = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', max_depth = None, min_samples_leaf = 1)
        rfc.fit(reduced_X_train, y_train)
        print("Training done!")

    elif classifier == 'myclassifier3':
        print('Training ' + classifier)

        #define features to be used in the classifier and reduce feature space
        classifier_features = ["urllink", "usermention", "hashtag", "elongatedword", "emoji", "laughtext", "allcaps", "ellipsis", "questionexclaim", "positiveword", "negativeword", "positiveface", "negativeface"]
        feature_columns_trainset = [mapping_train[feature] for feature in classifier_features]
        reduced_X_train = X_train[:, feature_columns_trainset]

        #add columns for lexicon scores and token counts
        reduced_X_train = np.append(reduced_X_train, lexicon_scores_train, axis = 1)
        reduced_X_train = np.append(reduced_X_train, token_count_train, axis = 1)

        #generate random forest classifier and fit training data
        rfc = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', max_depth = 13, min_samples_leaf = 10)
        rfc.fit(reduced_X_train, y_train)
        print("Training done!")

    for testset in testsets.testsets:

        #open testset
        testingTweets = []
        with open(testset, "r") as ts:
            for line in ts:
                tweet = line.strip().split("\t")
                testingTweets.append(tweet)

        #generate feature space for the testset
        testingIDs, mapping_test, X_test, lexicon_scores_test, token_count_test, y_test = get_feature_space(testingTweets, training = False)

        #reduce feature space as to match the training data
        feature_columns_testset = [mapping_test[feature] for feature in classifier_features]
        reduced_X_test = X_test[:, feature_columns_testset]
        reduced_X_test = np.append(reduced_X_test, lexicon_scores_test, axis = 1)
        reduced_X_test = np.append(reduced_X_test, token_count_test, axis = 1)

        #predict class labels for the testing data
        predicted_labels = rfc.predict(reduced_X_test)
        predictions = dict(zip(testingIDs, predicted_labels))

        #evaluate using the evaluation.py script provided
        evaluation.evaluate(predictions, testset, classifier)
        evaluation.confusion(predictions, testset, classifier)
