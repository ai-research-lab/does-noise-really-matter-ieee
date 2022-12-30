import jsonlines
import nltk


def load_nq(path):
    NQ = []
    num = sum(1 for line in open(path))
    with jsonlines.open(path) as f:
        for line in f.iter():
            NQ.append(line)
    return NQ


def short_answer(example, train=True):
    if train:
        short_ans = example['annotations'][0]['short_answers']
        if len(short_ans) == 0:
            return []
        else:
            spans = []
            for answer in short_ans:
                spans.append((answer['start_token'], answer['end_token']))
        return spans


def long_answer(example):
    long_ans = example['annotations'][0]['long_answer']
    return long_ans['start_token'], long_ans['end_token']


def nq_get_triplets(nq: list, train=True):
    def html_remove(x):
        from bs4 import BeautifulSoup
        return BeautifulSoup(x).get_text()

    tokenizer = nltk.WhitespaceTokenizer()
    questions = []
    answers = []
    contexts = []
    for i in range(len(nq)):
        example = nq[i]
        question = example['question_text']
        if train:
            text = example['document_text']
        elif not train:
            text = example['document_html']
        text_tokens = tokenizer.tokenize(text)
        answer_spans = short_answer(example)
        long_answer_span = long_answer(example)
        if long_answer_span[0] != -1:
            long_ans_as_context = html_remove(" ".join(text_tokens[long_answer_span[0]:long_answer_span[-1]]))
            for (start, end) in answer_spans:
                questions.append(question)
                contexts.append(long_ans_as_context)
                answers.append(html_remove(" ".join(text_tokens[start:end])))

    return questions, contexts, answers
