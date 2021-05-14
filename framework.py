import torch
import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
from utils import get_tokenizer
from data_loader import get_context_question

import numpy as np
import json
import time

tokenizer = get_tokenizer('/home/ubuntu/pycharm_proj/pretrained_models/bert/bert-base-cased-vocab.txt')

# todo: 取消在question中的tag，提高速度，删除单独的只有一个字母object


class Framework(object):
    def __init__(self, con):
        self.config = con
        self.REL_NUM = self.config.rel_num

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.cuda()

        # define the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr=self.config.learning_rate)

        # whether use multi GPU
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def loss(gold, pred, mask):
            pred = pred.squeeze(-1)
            los = F.binary_cross_entropy(pred, gold, reduction='none')
            if los.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            los = torch.sum(los * mask) / torch.sum(mask)
            return los

        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        # get the data loader
        print("Start loading train data.")
        train_data_loader = data_loader.get_loader(self.config, prefix=self.config.train_prefix)
        print("Start loading dev data.")
        dev_data_loader = data_loader.get_loader(self.config, prefix=self.config.dev_prefix, is_test=True)

        # other
        model.train()
        global_step = 0
        loss_sum = 0
        theta = self.config.theta

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()

            while data is not None:
                pred_sub_heads, pred_sub_tails, qa_outputs = model(data)

                sub_heads_loss = loss(data['sub_heads'], pred_sub_heads, data['mask'])
                sub_tails_loss = loss(data['sub_tails'], pred_sub_tails, data['mask'])

                qa_outputs_loss = loss(data['obj_qa_tags'], qa_outputs, data['qa_masks'])

                total_loss = theta * (sub_heads_loss + sub_tails_loss) + (1 - theta) * qa_outputs_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += total_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()

            if (epoch + 1) % self.config.test_epoch == 0 and epoch > 2:
                # start testing
                eval_start_time = time.time()
                model.eval()
                # call the test function
                self.logging("================Testing on dev================")
                sub_precision, sub_recall, sub_f1_score, precision, recall, f1_score = self.test(dev_data_loader, model)
                model.train()
                self.logging("Epoch {:3d}, eval time: {:5.2f}s \nsubject@ f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}"
                             .format(epoch, time.time() - eval_start_time, sub_f1_score, sub_precision, sub_recall))
                self.logging('triple@ f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}'
                             .format(f1_score, precision, recall))

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall

                    self.logging("Current best epoch: {:3d}, best f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}"
                                 .format(best_epoch, best_f1_score, best_precision, best_recall))
                    # save the best model
                    best_save_name = self.config.model_save_name \
                                                  + "_epoch_%d_f1_%.4f.pickle" % (epoch, best_f1_score)
                    path = os.path.join(self.config.checkpoint_dir, best_save_name)

                    self.logging("Saving the model as" + path)
                    torch.save(model.module.state_dict(), path)
                else:
                    self.logging("Current best epoch: {:3d}, best f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}"
                                 .format(best_epoch, best_f1_score, best_precision, best_recall))


            # manually release the unused cache
            torch.cuda.empty_cache()

        self.logging("finish training")
        self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
                     format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))

    def test(self, test_data_loader, model, output=False, h_bar=0.5, t_bar=0.5):
        # todo: parallel testing

        if output:
            # check the result dir
            if not os.path.exists(self.config.result_dir):
                os.mkdir(self.config.result_dir)

            path = os.path.join(self.config.result_dir, self.config.result_save_name)

            fw = open(path, 'w')

        orders = ['subject', 'relation', 'object']

        def to_tup(triple_list):
            ret = []
            for triple in triple_list:
                ret.append(tuple(triple))
            return ret

        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        # todo: 在此处所有data load完毕？
        id2rel = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[0]
        correct_num, predict_num, gold_num = 0, 0, 0
        sub_correct_num, sub_predict_num, sub_gold_num = 0, 0, 0

        while data is not None:
            with torch.no_grad():
                token_ids = data['token_ids']
                tokens = data['tokens'][0]
                mask = data['mask']
                text = data['qa_tokens']

                # print("test@text: ", text) todo: why print all texts?
                # get subjects
                if hasattr(model, 'module'):
                    encoded_text = model.module.get_encoded_text(token_ids, mask)
                    pred_sub_heads, pred_sub_tails = model.module.get_subs(encoded_text)
                else:
                    encoded_text = model.get_encoded_text(token_ids, mask)
                    pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)

                # select idx that pred > bar
                sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], \
                                       np.where(pred_sub_tails.cpu()[0] > t_bar)[0]
                subjects = []
                for sub_head in sub_heads:
                    sub_tail = sub_tails[sub_tails >= sub_head]
                    if len(sub_tail) > 0:
                        sub_tail = sub_tail[0]
                        subject = tokens[sub_head: sub_tail]
                        subjects.append((subject, sub_head, sub_tail))

                sub_list = []
                triple_list = []
                if subjects:
                    # get subjects text
                    for subject_idx, subject in enumerate(subjects):
                        sub = subject[0]
                        sub = ''.join([i.lstrip("##") for i in sub])
                        sub = ' '.join(sub.split('[unused1]'))
                        sub_list.append(sub)

                    # 对每个subject进行预测，问题生成
                    for sub_text in sub_list:
                        # print("test@text:", text[0])
                        # print("===================================test@sub:", sub_text)
                        if len(sub_text) > 20 or len(sub_text) <= 1:
                            continue

                        # subject correctly identified
                        for i in range(self.REL_NUM):
                            rel_text = id2rel[str(int(i))]

                            qa_token_ids, qa_masks, qa_tokens, qa_text_len = \
                                get_context_question(text[0], tokens, [sub_text], rel_text, self.config)

                            # print("test@qa:", qa_tokens)

                            qa_token_ids = torch.from_numpy(qa_token_ids).unsqueeze(0).cuda()
                            qa_masks = torch.from_numpy(qa_masks).unsqueeze(0).cuda()

                            # print("test@qa: ", type(qa_token_ids), "\n", qa_masks)
                            if hasattr(model, 'module'):
                                qa_encoded_text = model.module.get_encoded_text(qa_token_ids, qa_masks)
                                # [batch_size, seq_len, rel_num]
                                qa_outputs = model.module.get_objs_for_specific_sub(qa_encoded_text)
                            else:
                                qa_encoded_text = model.get_encoded_text(qa_token_ids, qa_masks)
                                # [batch_size, seq_len, rel_num]
                                qa_outputs = model.get_objs_for_specific_sub(qa_encoded_text)
                            # todo: get obj from qa_outputs
                            # =================ver1.0
                            # start_logits, end_logits = qa_outputs.split(1, dim=-1)
                            # # [batch_size, seq_len]
                            # start_logits = start_logits.squeeze(-1)
                            # # [batch_size, seq_len]
                            # end_logits = end_logits.squeeze(-1)
                            # # print("test@start_logits:", start_logits)
                            # # print("test@end_logits:", end_logits)
                            # obj_heads, obj_tails = np.where(start_logits[0].cpu()[0] > h_bar), \
                            #                        np.where(end_logits[0].cpu()[0] > t_bar)
                            # no object is flagged
                            # if len(obj_heads[0]) == 0 or len(obj_tails[0]) == 0:
                            #     continue
                            # for obj_head in zip(*obj_heads):
                            #     for obj_tail in zip(*obj_tails):
                            #         if obj_head <= obj_tail:
                            #             obj = tokens[obj_head: obj_tail]
                            #             obj = ''.join([i.lstrip("##") for i in obj])
                            #             obj = ' '.join(obj.split('[unused1]'))
                            #             triple_list.append((sub_text, rel_text, obj))
                            #             print("test@triple:", sub_text, rel_text, obj)
                            #             break

                            # =================ver2.0
                            qa_tag_idx = torch.argmax(qa_outputs, dim=-1).squeeze(dim=0)

                            obj_heads = torch.where(qa_tag_idx == 0)[0]
                            # print("test@obj", obj_heads[0], obj_tails[0])
                            if len(obj_heads) == 0:
                                continue
                            # print("test@qa_tag_idx:", qa_tag_idx)
                            # print("test@obj", obj_heads)
                            for obj_head in obj_heads:
                                obj_tail = int(obj_head) + 1
                                while int(qa_tag_idx[obj_tail]) == 1 and obj_tail < qa_text_len:
                                    obj_tail += 1
                                obj = tokens[obj_head: obj_tail]
                                # print("test@objs", obj)
                                obj = ''.join([i.lstrip("##") for i in obj])
                                obj = ' '.join(obj.split('[unused1]'))
                                if obj == ' ' or obj == '':
                                    continue
                                if obj[-1] == ' ':
                                    obj = obj[:-1]
                                triple_list.append((sub_text, rel_text, obj))
                                # print("test@triple:", sub_text, rel_text, obj, "**")
                                # break

                    triple_set = set()
                    for s, r, o in triple_list:
                        triple_set.add((s, r, o))
                    pred_list = list(triple_set)
                else:
                    pred_list = []

                pred_triples = set(pred_list)
                gold_triples = set(to_tup(data['triples'][0]))
                pred_subjects = set(sub_list)
                gold_subjects = set([tri[0] for tri in gold_triples])

                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                sub_correct_num += len(pred_subjects & gold_subjects)
                sub_predict_num += len(pred_subjects)
                sub_gold_num += len(gold_subjects)

                if output:
                    result = json.dumps({
                        # 'text': ' '.join(tokens),
                        'triple_list_gold': [
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]
                    }, ensure_ascii=False)
                    fw.write(result + '\n')

                data = test_data_prefetcher.next()

        print("subjects@ correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".
              format(sub_correct_num, sub_predict_num, sub_gold_num))
        print("triple@ correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        sub_precision = sub_correct_num / (sub_predict_num + 1e-10)
        sub_recall = sub_correct_num / (sub_gold_num + 1e-10)
        sub_f1_score = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-10)

        return sub_precision, sub_recall, sub_f1_score, precision, recall, f1_score

    def testall(self, model_pattern):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.test_model_name)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        sub_precision, sub_recall, sub_f1_score, precision, recall, f1_score = self.test(test_data_loader, model, True)
        print("subject@ f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(sub_f1_score, sub_precision, sub_recall))
        print("triple@ f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(f1_score, precision, recall))

