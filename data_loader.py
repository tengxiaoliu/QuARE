from torch.utils.data import DataLoader, Dataset
import json
import os
# from pytorch_pretrained_bert import BertTokenizer
import torch
from utils import get_tokenizer
import numpy as np
from random import choice

tokenizer = get_tokenizer('/home/ubuntu/pycharm_proj/pretrained_models/bert/bert-base-cased-vocab.txt')

"""
对于每一个句子，已经提取出subject->relation/object的对应map
"""
BERT_MAX_LEN = 512
QA_BERT_MAX_LEN = 512


# todo: maybe chunk max_len-20 in advance to save space for questions

# def get_question(tokens, rel):
#     """
#     Get question from tokens and form a question
#     :param tokens: list of tokenized texts
#     :return: context + question token_ids
#     """
#
#     qst = " Find the entities that have [relation] " + rel_text + " with [subject] " + sub_text
#     q_tokenized = tokenizer.tokenize(qst)
#     return tokens + s


def get_context_question(text, tokens, sub_text, rel_text, config, template=False):
    """
    combine context, question, and tokenize
    :return: context + question tokens, within BERT_MAX_LEN
    """

    sub = ''.join([i.lstrip("##") for i in sub_text[0]])
    sub_text = ' '.join(sub.split('[unused1]'))

    if template:
        json_tmpl = json.load(open(os.path.join(config.data_path + 'question_template.json')))
        rel_id = json_tmpl[1][rel_text]
        question = json_tmpl[2][rel_id]
        question = question.replace("XXX", sub_text)
    else:
        question = "Find the entities that have relation " + rel_text + " with subject " + sub_text

    # todo: debug
    # print("DL@qst: ", question)

    # ctx_qst_text = text + question
    # ctx_qst_text = ' '.join(ctx_qst_text.split()[:config.max_qa_len])
    qa_text = text + " " + question
    qst_tokens = tokenizer.tokenize(question)
    ctx_qst_tokens = tokens[:-1] + qst_tokens[1:]

    # print("qa_tokens@", ctx_qst_tokens)
    # print("qa_text@ ", qa_text)
    # tokenizer separate by "unused1": indicate blank space in sentences (boundaries of entities)

    if len(ctx_qst_tokens) > QA_BERT_MAX_LEN:
        ctx_qst_tokens = ctx_qst_tokens[: QA_BERT_MAX_LEN]

    ctx_qst_text_len = len(ctx_qst_tokens)

    token_ids, segment_ids = tokenizer.encode(first=qa_text)
    masks = segment_ids
    if len(token_ids) > ctx_qst_text_len:
        token_ids = token_ids[:ctx_qst_text_len]
        masks = masks[:ctx_qst_text_len]
    token_ids = np.array(token_ids)
    masks = np.array(masks) + 1

    return token_ids, masks, ctx_qst_tokens, ctx_qst_text_len


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class MyDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer
        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:500]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        # print("Loaded {} data from {}.json".format(str(len(self.json_data)), prefix))
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]
        self.id2rel = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[0]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        """
        get
        :param idx:
        :return:
        todo: how about negative sampling?
        """

        NEG = choice([1,1,1,1,1,0])

        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        text = ' '.join(text.split()[:self.config.max_len])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[: BERT_MAX_LEN]
        text_len = len(tokens)  # chunk -> tokenize -> chunk

        if not self.is_test:
            s2ro_map = {}  # (sub_head_idx, sub_tail_idx) : (obj_head_idx, obj_tail_idx, rel_id)
            sr2o_map = {}  # (sub_head_idx, sub_tail_idx, rel_id) : (obj_head_idx, obj_tail_idx)
            sub_id2text_map = {}
            for triple in ins_json_data['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    sub_rel = (sub_head_idx, sub_head_idx + len(triple[0]) - 1, self.rel2id[triple[1]])
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                        sub_id2text_map[sub] = []
                    if sub_rel not in sr2o_map:
                        sr2o_map[sub_rel] = []
                    # map stores object index and relation id
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))
                    sub_id2text_map[sub].append(triple[0])
                    sr2o_map[sub_rel].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1))

            if s2ro_map:
                token_ids, segment_ids = self.tokenizer.encode(first=text)
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1  # each element in mask +1
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)

                for s in s2ro_map:
                    sub_heads[s[0]] = 1  # s[0]: sub_head_idx
                    sub_tails[s[1]] = 1  # s[1]: sub_tail_idx
                # sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head_idx, sub_tail_idx, rel_id = choice(list(sr2o_map.keys()))

                # find a negative rel_id
                if NEG:
                    pos_rel_id = set()
                    for ro in s2ro_map.get((sub_head_idx, sub_tail_idx)):
                        pos_rel_id.add(ro[2])
                    rel_id = choice(list(set([i for i in range(24)])-pos_rel_id))

                # 一次数据只选取一个gold subject，以及和这个subject对应的object/relation标签
                # 在此处生成问题 context+question
                sub_text = sub_id2text_map.get((sub_head_idx, sub_tail_idx))
                # 与该sub对应的ro list (可能是1项或多项)
                # ro_list = s2ro_map.get((sub_head_idx, sub_tail_idx))
                # 与该sub,rel对应的obj list (可能是1项或多项) (obj_head_idx, obj_tail_idx)
                obj_list = sr2o_map.get((sub_head_idx, sub_tail_idx, rel_id))
                # randomly select a gold subject

                rel_text = self.id2rel[str(rel_id)]
                # print("dataloader: ", rel_id, rel_text)

                # get context + question (with selected subject and relation) tokens
                qa_token_ids, qa_masks, qa_tokens, qa_text_len = \
                    get_context_question(text, tokens, sub_text, rel_text, self.config)
                # chunk -> tokenize -> chunk

                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                # size: text_len, num_labels (each column is a sentence)
                obj_qa_tags = np.zeros((qa_text_len, self.config.num_labels))

                if not NEG: # positive sample
                    for obj in obj_list:
                        obj_qa_tags[obj[0]][0] = 1  # start tag
                        obj_qa_tags[obj[1]][1] = 1  # end tag following huggingface

                # obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)),
                # np.zeros((text_len, self.config.rel_num))

                # for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                #     obj_qa_tags[ro[0]][1] = 1  # start tag
                #     obj_qa_tags[ro[1]][3] = 1  # end tag following huggingface

                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, rel_id, qa_token_ids, \
                       qa_masks, qa_text_len, obj_qa_tags, ins_json_data['triple_list'], tokens, qa_tokens
            else:
                return None
        else:  # istest
            token_ids, segment_ids = self.tokenizer.encode(first=text)
            masks = segment_ids
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1  # 0->1, 1->2 ?
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            # todo: 怎样构造测试tensor？
            rel_id = 0
            qa_token_ids, qa_masks, qa_tokens, qa_text_len = token_ids, masks, tokens, text_len
            obj_qa_tags = np.zeros((qa_text_len, self.config.num_labels))
            # obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)),
            # np.zeros((text_len, self.config.rel_num))
            # return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
            #        ins_json_data['triple_list'], tokens
            # for the convenience of testing
            qa_tokens = text
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, rel_id, qa_token_ids, \
                   qa_masks, qa_text_len, obj_qa_tags, ins_json_data['triple_list'], tokens, qa_tokens


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)

    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, rel_id, qa_token_ids, \
    qa_masks, qa_text_len, obj_qa_tags, triples, tokens, qa_tokens = zip(*batch)

    # token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(
    #     *batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    max_qa_text_len = max(qa_text_len)

    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()

    batch_qa_token_ids = torch.LongTensor(cur_batch, max_qa_text_len).zero_()
    batch_qa_masks = torch.LongTensor(cur_batch, max_qa_text_len).zero_()
    batch_obj_qa_tags = torch.Tensor(cur_batch, max_qa_text_len, 2).zero_()  # self.config.num_labels

    # batch_obj_heads = torch.Tensor(cur_batch, max_text_len, 24).zero_()
    # batch_obj_tails = torch.Tensor(cur_batch, max_text_len, 24).zero_()

    for i in range(cur_batch):
        # wrap all into batch
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
        batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
        batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))

        batch_qa_token_ids[i, :qa_text_len[i]].copy_(torch.from_numpy(qa_token_ids[i]))
        batch_qa_masks[i, :qa_text_len[i]].copy_(torch.from_numpy(qa_masks[i]))
        batch_obj_qa_tags[i, :qa_text_len[i], :].copy_(torch.from_numpy(obj_qa_tags[i]))
        # batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
        # batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'sub_heads': batch_sub_heads,
            'sub_tails': batch_sub_tails,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'qa_token_ids': batch_qa_token_ids,
            'qa_masks': batch_qa_masks,
            'obj_qa_tags': batch_obj_qa_tags,
            # 'obj_heads': batch_obj_heads,
            # 'obj_tails': batch_obj_tails,
            'triples': triples,
            'tokens': tokens,
            'qa_tokens': qa_tokens}


def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=my_collate_fn):
    dataset = MyDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
        print("Loading {} data from {}.json".format(str(dataset.__len__()), prefix))
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
        print("Loading {} data from {}.json".format(str(dataset.__len__()), prefix))
    return data_loader


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
