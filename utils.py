from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import re
import datetime
import pickle
import random

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def data_loader(file_train, file_test):
    print("data_loader start!")

    data_list_train = []
    data_list_test = []
    data_idx_list_train = []
    data_idx_list_test = []
    max_story_len = 0
    max_words_len = 0

    rdic = []
    rdic.append("<nil>")

    # read file for train
    with open(file_train) as f:
        lines_train = f.readlines()

    for line_train in lines_train:
        line = line_train.lower()

        idx, line = line.split(" ", 1)
        idx = int(idx)

        #words = tokenize(line)
        #idx = int(words[0])
        #words = words[1:]

        if idx == 1:
            stories = []

        if "\t" in line:
            que, ans, sup = line.split('\t')
            que = tokenize(que)
            if que[-1] == "?":
                que = que[:-1]

            #que = words[:-3]
            #ans = words[-2]
            #sup = words[-1]
            data_list_train.append([stories, que, ans])
            stories = stories.copy()

            max_story_len = max(max_story_len, len(stories))
            max_words_len = max(max_words_len, len(que))

            for q in que:
                if q not in rdic:
                    rdic.append(q)

            if ans not in rdic:
                rdic.append(ans)
        else:
            #sen = words[:-1]
            sen = tokenize(line)
            if sen[-1] == ".":
                sen = sen[:-1]
            stories.append(sen)

            max_words_len = max(max_words_len, len(sen))

            for s in sen:
                if s not in rdic:
                    rdic.append(s)

    # read file for test
    with open(file_test) as f:
        lines_test = f.readlines()

    for line_test in lines_test:
        line = line_test.lower()

        idx, line = line.split(" ", 1)
        idx = int(idx)

        if idx == 1:
            stories = []

        if "\t" in line:
            que, ans, sup = line.split('\t')
            que = tokenize(que)
            if que[-1] == "?":
                que = que[:-1]

            data_list_test.append([stories, que, ans])
            stories = stories.copy()

            max_story_len = max(max_story_len, len(stories))
            max_words_len = max(max_words_len, len(que))

            for q in que:
                if q not in rdic:
                    rdic.append(q)

            if ans not in rdic:
                rdic.append(ans)
        else:
            sen = tokenize(line)
            if sen[-1] == ".":
                sen = sen[:-1]
            stories.append(sen)

            max_words_len = max(max_words_len, len(sen))

            for s in sen:
                if s not in rdic:
                    rdic.append(s)

    # make dic
    dic = {w:i for i,w in enumerate(rdic)}
    max_len = [max_story_len, max_words_len]

    # make dic[idx] list for train
    for data in data_list_train:
        stories = data[0]
        question = data[1]
        answer = data[2]

        stories_idx = []
        for story in stories:
            story_idx = [dic[w] for w in story]
            stories_idx.append(story_idx)

        question_idx = [dic[q] for q in question]
        answer_idx = dic[answer]

        data_idx_list_train.append([stories_idx, question_idx, answer_idx])

    # make dic[idx] list for test
    for data in data_list_test:
        stories = data[0]
        question = data[1]
        answer = data[2]

        stories_idx = []
        for story in stories:
            story_idx = [dic[w] for w in story]
            stories_idx.append(story_idx)

        question_idx = [dic[q] for q in question]

        answer_idx = dic[answer]

        data_idx_list_test.append([stories_idx, question_idx, answer_idx])

    print("data_loader end!")
    print()

    return data_list_train, data_list_test, data_idx_list_train, data_idx_list_test, rdic, dic, max_len