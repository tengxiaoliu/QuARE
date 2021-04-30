import torch
from transformers import *

def build_question(sub_text, rel_text, special_tag=False):
    if special_tag:
        s = "Find the object that has [relation] " + rel_text + " with [subject] " + sub_text
    else:
        s = "Find the object that has relation " + rel_text + " with subject " + sub_text
    return s

