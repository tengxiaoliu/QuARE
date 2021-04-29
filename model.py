from torch import nn
from transformers import *
import torch


class QAre(nn.Module):
    def __init__(self, config):
        super(QAre, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.bert_encoder = BertModel.from_pretrained(self.config.bert_dir)
        self.num_labels = config.num_labels

        # Subject tagger
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)

        # Object and relation QA module
        # todo: design QA module from huggingface

        self.tag_linear = nn.Linear(self.bert_encoder.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_func = nn.CrossEntropyLoss()
        self.theta = config.theta

    def get_subs(self, encoded_text):
        """
        Subject Taggers
        :param encoded_text: input sentence encoded by BERT
        :return: predicted subject head, predicted subject tail
        """
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    def get_objs_given_sub(self, sub):
        """
        Given tagged subject, form a question, and get answer span.
        todo: QA style to get answer span
        :param sub:
        :return: predicted object head, predicted object tail
        """




    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        """
        Relation-specific Object Taggers
        :param sub_head_mapping: [0,0,0,1,0,0,0]?
        :param sub_tail_mapping: [0,0,0,0,1,0,0]?
        :param encoded_text: input sentence pretrained with BERT
        :return: predicted object head, predicted object tail
        """
        # [batch_size, 1, bert_dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2
        # [batch_size, seq_len, bert_dim]

        # TODO: also add relation information into encoded text, will it help?
        # relation encoder?
        encoded_text = encoded_text + sub
        # [batch_size, seq_len, rel_num]
        pred_obj_heads = self.obj_heads_linear(encoded_text)
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        # [batch_size, seq_len, rel_num]
        pred_obj_tails = self.obj_tails_linear(encoded_text)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
        return pred_obj_heads, pred_obj_tails

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        # print("Inside get_encoded_text")
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size, seq_len, 1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        # [batch_size, 1, seq_len]
        sub_head_mapping = data['sub_head'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)

        # get question using sub_head, sub_tail


        #concatenate ori_sent embedding and formed question embedding?
        # return the index of object span from ori_sent


        # [batch_size, seq_len, rel_num]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails

    def get_question(self, sub_head, sub_tail):
        """
        Get text question given ground truth subject index and (for each) relation
        :param sub_head:
        :param sub_tail:
        :return: a text question given ground truth subject and tail
        if a corresponding relation doesn't contain an object in sentence, all tag should be 0
        """