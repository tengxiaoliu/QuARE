from torch import nn
from transformers import *
import torch


class QAre(nn.Module):
    def __init__(self, config):
        super(QAre, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.bert = BertModel.from_pretrained(self.config.bert_dir)
        self.num_labels = config.num_labels

        # Subject tagger
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)

        # Object and relation QA module
        # todo: design QA module from huggingface

        self.qa_linear = nn.Linear(self.bert_dim, self.num_labels)
        # self.dropout = nn.Dropout(config.dropout_prob)
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

    def get_objs_given_sub(self,
                           input_ids=None,
                           attention_mask=None,
                           token_type_ids=None,
                           position_ids=None,
                           head_mask=None,
                           inputs_embeds=None,
                           start_positions=None,
                           end_positions=None,
                           output_attentions=None,
                           output_hidden_states=None,
                           ):
        """
        Given tagged subject, form a question, and get answer span.
        todo: QA style to get answer span
        After tokenizer：input_ids，token_type_ids，attention_mask
        :param input_ids: from tokenizer
        :param attention_mask: whether apply attention
        :param token_type_ids: question 0, context 1
        :param position_ids:
        :param head_mask:
        :param inputs_embeds:
        :param start_positions: torch.tensor([start_tag])
        :param end_positions: torch.tensor([end_tag])
        :param output_attentions:
        :param output_hidden_states:
        :return: object prediction loss
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = self.qa_linear(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = self.loss_func(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

    def get_objs_for_specific_sub(self, qa_encoded_text):
        """
        Relation-specific Object Taggers
        :param sub_head_mapping: [0,0,0,1,0,0,0]?
        :param sub_tail_mapping: [0,0,0,0,1,0,0]?
        :param encoded_text: input sentence pretrained with BERT
        :return: predicted object head, predicted object tail
        """
        # # [batch_size, 1, bert_dim]
        # sub_head = torch.matmul(sub_head_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        # # [batch_size, 1, bert_dim]
        # sub = (sub_head + sub_tail) / 2
        # # [batch_size, seq_len, bert_dim]

        # create embeddings for ctx_qst, given subject and relation

        # encoded_text should be concatenation of context and question
        qa_outputs = self.qa_linear(qa_encoded_text)
        # [batch_size, seq_len, label_num]
        qa_outputs = torch.sigmoid(qa_outputs)
        # todo: extract pred_obj_heads, pred_obj_tails from qa_outputs
        return qa_outputs

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        # print("Inside get_encoded_text")
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
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

        # [batch_size, seq_len]
        qa_token_ids = data['qa_token_ids']
        # [batch_size, seq_len]
        qa_mask = data['qa_mask']
        # [batch_size, seq_len, bert_dim(768)]
        qa_encoded_text = self.get_encoded_text(qa_token_ids, qa_mask)
        # [batch_size, seq_len, rel_num]
        qa_outputs = self.get_objs_for_specific_sub(qa_encoded_text)

        # [batch_size, seq_len, rel_num]
        # pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(encoded_text)
        return pred_sub_heads, pred_sub_tails, qa_outputs

    def get_question(self, sub_head, sub_tail):
        """
        Get text question given ground truth subject index and (for each) relation
        :param sub_head:
        :param sub_tail:
        :return: a text question given ground truth subject and tail
        if a corresponding relation doesn't contain an object in sentence, all tag should be 0
        """