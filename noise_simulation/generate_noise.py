import argparse
import pandas as pd
import numpy as np
import os
import nltk
from copy import deepcopy
import re
import json
from transformers import BertTokenizer
from tokenizers import decoders
import logging

logger = logging.getLogger(__name__)


def train_dev_split(args, train_set: pd.core.frame.DataFrame, negative_examples=False):
    # dev_set: pd.core.frame.DataFrame,
    """
    Cuts DataFrames into reqires sizes, creates holdout_dataset for noise resource

    Arguments:
    train_set -- pandas DataFrame with training data
    dev_set -- pandas DataFrame with development data
    negative_examples -- Set False to delete negative examples
    train_set_size -- int; Size of training set
    dev_ratio -- float; Development set size (ratio to train set size)

    Return:
    train - pd.Dataframe of shape (train_set_size, len(train_set.columns))
    holdout_train - pd.Dataframe of shape (train_set.shape[0] - train_set_size, len(train_set.columns))
    dev - pd.Dataframe of shape (train_set_size*dev_ratio//1, len(dev_set.columns))
    """
    # delete negative examples if negative_examples=False
    if not negative_examples:
        train_set = train_set.drop(
            train_set[train_set.isna().any(axis=1)].index
        ).reset_index(drop=True)

    # cut datasets
    train = train_set.iloc[: args.train_set_size].reset_index(drop=True)
    holdout_train = train_set.iloc[args.train_set_size :].reset_index(drop=True)
    return train, holdout_train


def train_holdout_split_random_indices(train_set: pd.core.frame.DataFrame, args):
    """
    Create DataFrames of reqired sizes, creates holdout_dataset for noise resource
    Used in SQuAD 1.1 dataset reduction as cutting can bias the dataset with topics concenration

    Arguments:
    train_set -- pandas DataFrame with training data
    random_seed -- seed for indices
    train_set_size -- int; Size of training set

    Return:
    train - pd.Dataframe of shape (train_set_size, len(train_set.columns))
    holdout_train - pd.Dataframe of shape (train_set.shape[0] - train_set_size, len(train_set.columns))
    """
    np.random.seed(42)  # args.random_seed
    ind_train = np.random.choice(
        range(train_set.shape[0]), size=args.train_set_size, replace=False
    )
    # cut datasets
    train = train_set.iloc[ind_train].reset_index(drop=True)
    holdout_train = train_set.iloc[
        np.setdiff1d(range(train_set.shape[0]), ind_train, assume_unique=True)
    ].reset_index(drop=True)

    return train, holdout_train


def cqa(args, dataset: pd.core.frame.DataFrame, start_of_search=0):
    """
    Creates lists of contexts, questions and answers dictionaries
    Used in create_dict function

    Arguments:
    dataset -- pandas DataFrame

    Return:
    contexts - list of length=dataset.shape[0]
    questions - list of length=dataset.shape[0]
    answers - list of length=dataset.shape[0] with dict elements:
            {'answer_start': int (char number), 'text': str (span answer)},
    """
    if args.noise_type == "Q1-T1-A(last char)":
        start_of_search = -1
    data = dataset.copy()
    contexts = data.text.tolist()
    questions = data.question.tolist()
    anss = []
    for ind in data.index:
        if start_of_search < 0:
            start = len(data["text"][ind]) - 1
        else:
            start = 0
        end = len(data["text"][ind])
        if type(data["span_answer"][ind]) == str:
            ans_start = data["text"][ind].find(data["span_answer"][ind], start, end)
            if ans_start > 0:
                anss.append(ans_start)
            else:
                anss.append(data["text"][ind].find(data["span_answer"][ind], 0, end))
        else:
            None
    data["answer_start"] = pd.Series(anss).astype(int)
    answers = [
        {
            "answer_start": int(data["answer_start"][ind]),
            "text": data["span_answer"][ind],
        }
        for ind in data.index
    ]
    return contexts, questions, answers


def create_dict(
    args, dataset: pd.core.frame.DataFrame, no_goldanswers=True, start_of_search=0
):
    """
    Creates a dictionary for saving as json

    Arguments:
    dataset -- pandas DataFrame
    no_goldanswers -- set False if there is no gold answers in dataset (in training set e.g.)

    Return:
    dictionary - in a form required in Huggingface examples for QA
    """
    if args.noise_type == "Q1-T1-A(last char)":
        start_of_search = -1
    contexts, questions, answers = cqa(args, dataset, start_of_search)
    gold_answers = np.array(dataset.answer)
    dictionary = {}
    if no_goldanswers:
        dictionary["data"] = [
            {
                "title": "qa",
                "paragraphs": [
                    {
                        "context": contexts[i],
                        "qas": [
                            {
                                "answers": [answers[i]],
                                "question": questions[i],
                                "id": "%d" % i,
                            }
                        ],
                    }
                    for i in range(dataset.shape[0])
                ],
            }
        ]
    else:
        dictionary["data"] = [
            {
                "title": "qa",
                "paragraphs": [
                    {
                        "context": contexts[i],
                        "qas": [
                            {
                                "answers": [answers[i]]
                                + [{"answer_start": None, "text": gold_answers[i]}],
                                "question": questions[i],
                                "id": "%d" % i,
                            }
                        ],
                    }
                    for i in range(dataset.shape[0])
                ],
            }
        ]
    return dictionary


def json_to_pd(json_file):
    contexts = []
    questions = []
    answers = []
    for cont in json_file["data"]:
        for paragraph in cont["paragraphs"]:
            for qas in paragraph["qas"]:
                if len(qas["answers"]) > 0:
                    contexts.append(paragraph["context"])
                    questions.append(qas["question"])
                    answers.append(qas["answers"][0]["text"])

    df = pd.DataFrame(
        {
            "text": pd.Series(contexts),
            "question": pd.Series(questions),
            "span_answer": pd.Series(answers),
            "answer": pd.Series(answers),
        }
    )
    return df


# ## Noise insertion functions


def noise_insertion_functions_trainholdout(
    args, train: pd.core.frame.DataFrame, holdout: pd.core.frame.DataFrame
):
    train_copy = deepcopy(train)
    np.random.seed(args.random_seed)
    # indeces of train pd.DataFrame rows to change
    indeces_context = np.random.choice(
        np.arange(0, train.shape[0]),
        size=int(train.shape[0] * float(args.noise_ratio)),
        replace=False,
    )
    # indeces of holdout pd.DataFrame rows to use as noise
    indeces_noise = np.random.choice(
        np.arange(0, holdout.shape[0]),
        size=int(train.shape[0] * float(args.noise_ratio)),
        replace=False,
    )
    contexts_train, contexts_HO = np.array(train.text), np.array(holdout.text)
    questions_train, questions_HO = np.array(train.question), np.array(holdout.question)
    answers_train, answers_HO = (
        np.array(train.span_answer),
        np.array(holdout.span_answer),
    )
    gold_answers_train, gold_answers_HO = (
        np.array(train.answer),
        np.array(holdout.answer),
    )
    tokenizer = nltk.WhitespaceTokenizer()

    if args.noise_type == "Q1-T2-A(rand)":
        for index_noise, elem_index_context in enumerate(indeces_context):
            # replace correct passage with noise
            contexts_train[elem_index_context] = contexts_HO[indeces_noise[index_noise]]
            # normalization of whitespaces
            contexts_train[elem_index_context] = " ".join(
                tokenizer.tokenize(contexts_train[elem_index_context])
            )
            # passage tokenization
            answer_tokens = tokenizer.tokenize(contexts_train[elem_index_context])
            ind = np.random.choice(
                np.arange(0, len(answer_tokens)), size=2, replace=False
            )  # new span boundaries
            # replace corresponding answer with random span
            answers_train[elem_index_context] = " ".join(
                answer_tokens[min(ind) : max(ind)]
            )
            gold_answers_train[elem_index_context] = answers_train[elem_index_context]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif args.noise_type == "Q1-T1(with replaced span A2)-A2":
        for index_noise, elem_index_context in enumerate(indeces_context):
            start_char_train = contexts_train[elem_index_context].find(
                answers_train[elem_index_context]
            )
            start_char_HO = contexts_HO[indeces_noise[index_noise]].find(
                answers_HO[indeces_noise[index_noise]]
            )
            contexts_train[elem_index_context] = (
                contexts_train[elem_index_context][:start_char_train]
                + answers_HO[indeces_noise[index_noise]]
                + contexts_train[elem_index_context][
                    start_char_train + len(answers_train[elem_index_context]) :
                ]
            )
            answers_train[elem_index_context] = answers_HO[indeces_noise[index_noise]]
            gold_answers_train[elem_index_context] = answers_train[elem_index_context]

            train_copy.text = contexts_train
            train_copy.span_answer = answers_train
            train_copy.answer = gold_answers_train

    elif args.noise_type == "Q1-T2-A(rand_short)":
        for index_noise, elem_index_context in enumerate(indeces_context):
            # replace correct passage with noise
            contexts_train[elem_index_context] = contexts_HO[indeces_noise[index_noise]]
            # normalization of whitespaces
            contexts_train[elem_index_context] = " ".join(
                tokenizer.tokenize(contexts_train[elem_index_context])
            )
            # passage tokenization
            answer_tokens = tokenizer.tokenize(contexts_train[elem_index_context])
            ans_len = np.random.randint(1, 10)
            while len(answer_tokens) - ans_len <= 0:
                ans_len = np.random.randint(1, 10)
            ind = np.random.randint(0, len(answer_tokens) - ans_len)
            answers_train[elem_index_context] = " ".join(
                answer_tokens[ind : ind + ans_len]
            )
            gold_answers_train[elem_index_context] = answers_train[elem_index_context]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif args.noise_type == "Q1-(T1+Trand)-A1":
        for index_noise, elem_index_context in enumerate(indeces_context):
            # replace correct passage with noise
            contexts_train[elem_index_context] += contexts_HO[
                indeces_noise[index_noise]
            ]
            answers_train[elem_index_context] = answers_HO[indeces_noise[index_noise]]
            gold_answers_train[elem_index_context] = gold_answers_HO[
                indeces_noise[index_noise]
            ]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif args.noise_type == "Q1-T2-A2":
        for index_noise, elem_index_context in enumerate(indeces_context):
            # replace correct passage with noise, answers and GA with respect to context
            contexts_train[elem_index_context] = contexts_HO[indeces_noise[index_noise]]
            answers_train[elem_index_context] = answers_HO[indeces_noise[index_noise]]
            gold_answers_train[elem_index_context] = gold_answers_HO[
                indeces_noise[index_noise]
            ]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif (
        args.noise_type == "Q1-T2-A(last 5 tok)"
        or args.noise_type == "Q1-T2-A(first 5 tok)"
    ):
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer_bert.decoder = decoders.WordPiece()
        for index_noise, elem_index_noise in enumerate(indeces_noise):
            tokens = tokenizer_bert.tokenize(
                contexts_train[indeces_context[index_noise]]
            )
            context_ids = tokenizer_bert.encode(
                contexts_train[indeces_context[index_noise]]
            )[1:-1]
            contexts_train[indeces_context[index_noise]] = tokenizer_bert.decode(
                context_ids
            )

            if args.noise_type == "Q1-T2-A(last 5 tok)":
                answer_span = tokens[-5:]
                answer_ids = context_ids[-5:]
            elif args.noise_type == "Q1-T2-A(first 5 tok)":
                answer_span = tokens[:5]
                answer_ids = context_ids[:5]

            # replace corresponding answer with random span
            answers_train[indeces_context[index_noise]] = tokenizer_bert.decode(
                answer_ids
            )
            if "##" == answers_train[indeces_context[index_noise]][:2]:
                answers_train[indeces_context[index_noise]] = answers_train[
                    indeces_context[index_noise]
                ][2:]
            answers_train[indeces_context[index_noise]] = re.sub(
                re.compile("(\s+'|'\s+)"),
                "'",
                answers_train[indeces_context[index_noise]],
            )
            contexts_train[indeces_context[index_noise]] = re.sub(
                re.compile("(\s+'|'\s+)"),
                "'",
                contexts_train[indeces_context[index_noise]],
            )
            gold_answers_train[indeces_context[index_noise]] = answers_train[
                indeces_context[index_noise]
            ]
            questions_train[indeces_context[index_noise]] = questions_HO[
                elem_index_noise
            ]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.question = questions_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif (
        args.noise_type == "Q1-T2-A(last 5 tok) whitespace"
        or args.noise_type == "Q1-T2-A(first 5 tok) whitespace"
    ):
        for index_noise, elem_index_noise in enumerate(indeces_noise):
            # answer tokenization
            answer_tokens = tokenizer.tokenize(
                contexts_train[indeces_context[index_noise]]
            )

            # normalization of whitespaces
            contexts_train[indeces_context[index_noise]] = " ".join(
                tokenizer.tokenize(" ".join(answer_tokens))
            )
            # take required tokens
            if args.noise_type == "Q1-T2-A(last 5 tok) whitespace":
                answer_span = answer_tokens[-5:]
            elif args.noise_type == "Q1-T2-A(first 5 tok) whitespace":
                answer_span = answer_tokens[:5]
            # replace corresponding answer with random span
            answers_train[indeces_context[index_noise]] = " ".join(
                tokenizer.tokenize(" ".join(answer_span))
            )
            gold_answers_train[indeces_context[index_noise]] = answers_train[
                indeces_context[index_noise]
            ]
            questions_train[indeces_context[index_noise]] = questions_HO[
                elem_index_noise
            ]

        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.question = questions_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif args.noise_type == "Dataset reduction":
        train_copy = train_copy.drop(labels=indeces_context).reset_index(drop=True)

    return train_copy


def noise_insertion_functions_trainsetonly(args, train: pd.core.frame.DataFrame):
    train_copy = deepcopy(train)
    np.random.seed(args.random_seed)
    # indeces of pd.DataFrame rows to corrupt
    indeces_noise = np.random.choice(
        np.arange(0, train.shape[0]),
        size=int(train.shape[0] * float(args.noise_ratio)),
        replace=False,
    )

    contexts_train = np.array(train.text)
    questions_train = np.array(train.question)
    answers_train = np.array(train.span_answer)
    gold_answers_train = np.array(train.answer)
    tokenizer = nltk.WhitespaceTokenizer()

    if args.noise_type == "Q(empty)-T1-A(rand)":
        for elem_index_noise in indeces_noise:
            # normalization of whitespaces
            contexts_train[elem_index_noise] = " ".join(
                tokenizer.tokenize(contexts_train[elem_index_noise])
            )
            # passage tokenization
            answer_tokens = tokenizer.tokenize(contexts_train[elem_index_noise])
            ind = np.random.choice(
                np.arange(0, len(answer_tokens)), size=2, replace=False
            )  # new span boundaries
            # replace corresponding answer with random span
            answers_train[elem_index_noise] = " ".join(
                answer_tokens[min(ind) : max(ind)]
            )
            gold_answers_train[elem_index_noise] = answers_train[elem_index_noise]
            questions_train[elem_index_noise] = "?"

        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.question = questions_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif (
        args.noise_type == "Q1-T1-A(last 5 tok)"
        or args.noise_type == "Q1-T1-A(first 5 tok)"
    ):
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer_bert.decoder = decoders.WordPiece()
        for elem_index_noise in indeces_noise:
            tokens = tokenizer_bert.tokenize(contexts_train[elem_index_noise])
            context_ids = tokenizer_bert.encode(contexts_train[elem_index_noise])[1:-1]
            contexts_train[elem_index_noise] = tokenizer_bert.decode(context_ids)

            if args.noise_type == "Q1-T1-A(last 5 tok)":
                answer_span = tokens[-5:]
                answer_ids = context_ids[-5:]
            elif args.noise_type == "Q1-T1-A(first 5 tok)":
                answer_span = tokens[:5]
                answer_ids = context_ids[:5]

            # replace corresponding answer with random span
            answers_train[elem_index_noise] = tokenizer_bert.decode(answer_ids)
            if "##" == answers_train[elem_index_noise][:2]:
                answers_train[elem_index_noise] = answers_train[elem_index_noise][2:]
            answers_train[elem_index_noise] = re.sub(
                re.compile("(\s+'|'\s+)"), "'", answers_train[elem_index_noise]
            )
            contexts_train[elem_index_noise] = re.sub(
                re.compile("(\s+'|'\s+)"), "'", contexts_train[elem_index_noise]
            )
            gold_answers_train[elem_index_noise] = answers_train[elem_index_noise]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif (
        args.noise_type == "Q(from T1)-T1-A(last 5 tok)"
        or args.noise_type == "Q(from T1)-T1-A(first 5 tok)"
    ):
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer_bert.decoder = decoders.WordPiece()
        for elem_index_noise in indeces_noise:
            tokens = tokenizer_bert.tokenize(contexts_train[elem_index_noise])
            context_ids = tokenizer_bert.encode(contexts_train[elem_index_noise])[1:-1]
            contexts_train[elem_index_noise] = tokenizer_bert.decode(context_ids)
            question_length = np.random.randint(4, 8)
            question_start = np.random.randint(
                0, 1 + len(context_ids) - question_length
            )
            questions_train[elem_index_noise] = tokenizer_bert.decode(
                context_ids[question_start : question_start + question_length]
            )

            if args.noise_type == "Q(from T1)-T1-A(last 5 tok)":
                answer_span = tokens[-5:]
                answer_ids = context_ids[-5:]
            elif args.noise_type == "Q(from T1)-T1-A(first 5 tok)":
                answer_span = tokens[:5]
                answer_ids = context_ids[:5]

            answers_train[elem_index_noise] = tokenizer_bert.decode(answer_ids)
            if "##" == answers_train[elem_index_noise][:2]:
                answers_train[elem_index_noise] = answers_train[elem_index_noise][2:]
            answers_train[elem_index_noise] = re.sub(
                re.compile("(\s+'|'\s+)"), "'", answers_train[elem_index_noise]
            )
            contexts_train[elem_index_noise] = re.sub(
                re.compile("(\s+'|'\s+)"), "'", contexts_train[elem_index_noise]
            )
            gold_answers_train[elem_index_noise] = answers_train[elem_index_noise]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.question = questions_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif (
        args.noise_type == "Q1-T1-A(last 5 tok) whitespace"
        or args.noise_type == "Q1-T1-A(first 5 tok) whitespace"
    ):
        for elem_index_noise in indeces_noise:
            # answer tokenization
            answer_tokens = tokenizer.tokenize(contexts_train[elem_index_noise])
            # normalization of whitespaces
            contexts_train[elem_index_noise] = " ".join(
                tokenizer.tokenize(" ".join(answer_tokens))
            )
            # take required tokens
            if args.noise_type == "Q1-T1-A(last 5 tok) whitespace":
                answer_span = answer_tokens[-5:]
            elif args.noise_type == "Q1-T1-A(first 5 tok) whitespace":
                answer_span = answer_tokens[:5]
            # replace corresponding answer with random span
            answers_train[elem_index_noise] = " ".join(
                tokenizer.tokenize(" ".join(answer_span))
            )
            gold_answers_train[elem_index_noise] = answers_train[elem_index_noise]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif args.noise_type == "Q1-T1-A(last char)":
        print("for create_dict function start_of_search=-1")
        answers_train, contexts_train = (
            np.array(train.span_answer),
            np.array(train.text),
        )
        for elem_index_noise in indeces_noise:
            answers_train[elem_index_noise] = contexts_train[elem_index_noise][-1]
        # upload changes
        train_copy.span_answer = answers_train

    elif args.noise_type == "Q(empty)-T1-A1":
        for elem_index_noise in indeces_noise:
            questions_train[elem_index_noise] = "?"
        # upload changes
        train_copy.question = questions_train

    elif args.noise_type == "Q1-T1-A(rand)":
        for elem_index_noise in indeces_noise:
            # normalization of whitespaces
            contexts_train[elem_index_noise] = " ".join(
                tokenizer.tokenize(contexts_train[elem_index_noise])
            )
            # passage tokenization
            answer_tokens = tokenizer.tokenize(contexts_train[elem_index_noise])
            ind = np.random.choice(
                np.arange(0, len(answer_tokens)), size=2, replace=False
            )  # new span boundaries
            # replace corresponding answer with random span
            answers_train[elem_index_noise] = " ".join(
                answer_tokens[min(ind) : max(ind)]
            )
            gold_answers_train[elem_index_noise] = answers_train[elem_index_noise]
        # upload changes
        train_copy.text = contexts_train
        train_copy.span_answer = answers_train
        train_copy.answer = gold_answers_train  # useless as do not used in training

    elif args.noise_type == "Q1-T1-A1(in the end)":
        for elem_index_noise in indeces_noise:
            start_ans = contexts_train[elem_index_noise].find(
                answers_train[elem_index_noise]
            )
            contexts_train[elem_index_noise] = (
                contexts_train[elem_index_noise][:start_ans]
                + contexts_train[elem_index_noise][
                    start_ans + len(answers_train[elem_index_noise]) :
                ]
                + contexts_train[elem_index_noise][
                    start_ans : start_ans + len(answers_train[elem_index_noise])
                ]
            )
        # upload changes
        train_copy.text = contexts_train

    elif args.noise_type == "Dataset reduction":
        train_copy = train_copy.drop(labels=indeces_noise).reset_index(drop=True)

    return train_copy


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="Directory to JSON train file to insert a noise in",
    )

    parser.add_argument(
        "--train_output_dir",
        default=None,
        type=str,
        required=True,
        help="Output directory for train file",
    )

    parser.add_argument(
        "--noise_ratio",
        default=None,
        type=float,
        required=True,
        help="Rate of noise in train set",
    )

    parser.add_argument(
        "--noise_type",
        default="Q1-T1-A(rand)",
        type=str,
        required=True,
        help="Type of noise: \n\n available types for train set only: \
        \n Q(empty)-T1-A(rand), \n Q1-T1-A(last 5 tok) , \n Q1-T1-A(first 5 tok), \
        \n Q(from T1)-T1-A(last 5 tok), Q(from T1)-T1-A(first 5 tok), \
        \n Q1-T1-A(last 5 tok) whitespace, \n Q1-T1-A(first 5 tok) whitespace, \
        \n Q1-T1-A(last char), \n Q(empty)-T1-A1, \n Q1-T1-A(rand), \n Q1-T1-A1(in the end), \
        \n Dataset reduction \
        \n\n available types for train-holdout sets: \n Q1-T2-A(rand), \n Q1-T2-A(rand_short),\
        \n Q1-(T1+Trand)-A1, \n Q1-T2-A2, \n Q1-T1(with replaced span A2)-A2, \
        \n Q1-T2-A(last 5 tok), \n Q1-T2-A(first 5 tok), \n Q1-T2-A(last 5 tok) whitespace",
    )

    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=False,
        help="Name for output file",
    )

    parser.add_argument(
        "--train_set_size",
        default=30000,
        type=int,
        required=False,
        help="Required dataset size",
    )

    parser.add_argument(
        "--dev_ratio",
        default=0.2,
        type=float,
        required=False,
        help="Size of dev set as the ratio from train set size",
    )

    parser.add_argument(
        "--dev_from_holdout",
        action="store_true",
        help="If the dev.json file is too small",
    )

    parser.add_argument(
        "--dev_output_dir",
        default="",
        type=str,
        required=False,
        help="Output directory for dev file",
    )

    parser.add_argument(
        "--random_seed", default=42, type=int, required=False, help="Random seed",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # download *.JSON format train_file
    if args.train_file.endswith(".json"):
        logger.info("Downloading *.JSON train file")
        with open(args.train_file) as json_file:
            train_json = json.load(json_file)
        df = json_to_pd(train_json)
        # delete NaN values
        df = df.drop(df[df.isna().any(axis=1)].index).reset_index(drop=True)
        if df.shape[0] > args.train_set_size:
            train_pd, holdout_pd = train_holdout_split_random_indices(df, args)
            if args.dev_from_holdout:
                dev_pd = holdout_pd[
                    -int(args.train_set_size * args.dev_ratio) :
                ].reset_index(drop=True)
                holdout_pd = holdout_pd[
                    : -int(args.train_set_size * args.dev_ratio)
                ].reset_index(drop=True)
                # save dev file in directory if in args
                if args.dev_output_dir:
                    json_dict_dev = create_dict(args, dev_pd, no_goldanswers=False)
                    if not os.path.exists(
                        os.path.join(os.getcwd(), args.dev_output_dir)
                    ):
                        os.makedirs(os.path.join(os.getcwd(), args.dev_output_dir))
                    logger.info(
                        "Saving file into {}".format(
                            os.path.join(
                                os.getcwd(),
                                args.dev_output_dir,
                                "dev_{name}.json".format(name=args.dataset_name),
                            )
                        )
                    )
                    with open(
                        os.path.join(
                            os.getcwd(),
                            args.dev_output_dir,
                            "dev_{name}.json".format(name=args.dataset_name),
                        ),
                        "w",
                    ) as fp:
                        json.dump(json_dict_dev, fp)

        elif df.shape[0] == args.train_set_size:
            train_pd = df
        else:
            raise ValueError(
                "Number of train examples exceeds available number of samples \
                            (dataset_size = {real_train_set_size}, desired_size = {desired_train_set_size}).".format(
                    real_train_set_size=df.shape[0],
                    desired_train_set_size=args.train_set_size,
                )
            )

    # download *.CSV format train_file (MS_MARCO)
    elif args.train_file.endswith(".csv"):
        logger.info("Downloading *.CSV train file")
        df = pd.read_csv(args.train_file)
        if df.shape[0] > args.train_set_size:
            train_pd, holdout_pd = train_dev_split(args, df)

    if args.noise_type in [
        "Q(empty)-T1-A(rand)",
        "Q1-T1-A(last 5 tok)",
        "Q1-T1-A(first 5 tok)",
        "Q(from T1)-T1-A(last 5 tok)",
        "Q(from T1)-T1-A(first 5 tok)",
        "Q1-T1-A(last 5 tok) whitespace",
        "Q1-T1-A(first 5 tok) whitespace",
        "Q1-T1-A(last char)",
        "Q(empty)-T1-A1",
        "Q1-T1-A(rand)",
        "Dataset reduction",
        "Q1-T1-A1(in the end)",
    ]:
        logger.info(
            "Inserting %.2f %% %s noise" % (float(args.noise_ratio), args.noise_type)
        )
        result = noise_insertion_functions_trainsetonly(args, train_pd)
    elif (
        args.noise_type
        in [
            "Q1-T2-A(rand)",
            "Q1-T2-A(rand_short)",
            "Q1-(T1+Trand)-A1",
            "Q1-T2-A2",
            "Q1-T2-A(last 5 tok)",
            "Q1-T2-A(first 5 tok)",
            "Q1-T2-A(last 5 tok) whitespace",
            "Q1-T2-A(first 5 tok) whitespace",
            "Q1-T1(with replaced span A2)-A2",
        ]
    ) and (df.shape[0] > args.train_set_size):
        logger.info(
            "Inserting %.2f %% %s noise" % (float(args.noise_ratio), args.noise_type)
        )
        result = noise_insertion_functions_trainholdout(args, train_pd, holdout_pd)
    else:
        raise ValueError("Type of noise {} doesn't exist.".format(args.noise_type))

    logger.info("Creating file in SQuAD format")
    json_dict = create_dict(args, result, no_goldanswers=True)
    if not os.path.exists(os.path.join(os.getcwd(), args.train_output_dir)):
        os.makedirs(os.path.join(os.getcwd(), args.train_output_dir))
    logger.info(
        "Saving file into {}".format(
            os.path.join(
                os.getcwd(),
                args.train_output_dir,
                "train_{name}_{noise_rate}.json".format(
                    name=args.dataset_name, noise_rate=args.noise_ratio
                ),
            )
        )
    )
    with open(
        os.path.join(
            os.getcwd(),
            args.train_output_dir,
            "train_{name}_{noise_rate}.json".format(
                name=args.dataset_name, noise_rate=args.noise_ratio
            ),
        ),
        "w",
    ) as fp:
        json.dump(json_dict, fp)

    return json_dict


if __name__ == "__main__":
    main()
