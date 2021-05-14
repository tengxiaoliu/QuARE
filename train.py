import config
import framework
import argparse
import model
import os
import torch
import numpy as np
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='QAre', help='name of the model')
parser.add_argument('--bert_dir', type=str, default='bert-base-cased', help='direction of the pretrained model')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NYT')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--test_epoch', type=int, default=5)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--test_model_name', type=str, default='QAre_DATASET_NYT_LR_1e-05_BS_80510.pickle')
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--max_qa_len', type=int, default=170)
parser.add_argument('--rel_num', type=int, default=24)
parser.add_argument('--num_labels', type=int, default=3)  # BEO
parser.add_argument('--period', type=int, default=50, help='print loss per PERIOD global steps')
parser.add_argument('--seed', type=int, default=71)
parser.add_argument('--theta', type=float, default=0.5)
parser.add_argument('--neg_samp', type=int, default=5, help='ratio of negative sampling')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--gpus', type=str, default="0,1,2,3")
args = parser.parse_args()

for arg in vars(args):
    print(arg, ":",  getattr(args, arg))

con = config.Config(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


fw = framework.Framework(con)

model = {
    'QAre': model.QAre
}

fw.train(model[args.model_name])
