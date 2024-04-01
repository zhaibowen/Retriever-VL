import os
import re
import json
import copy
import random
import pickle
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast
from config import RetrieverVLConfig_medium

query_template = [
    "What is in the photo?",
    "Render a clear and concise summary of the photo.",
    "Give a brief description of the image.",
    "Provide a brief description of the given image.",
    "What is this?",
    "Write a terse but informative summary of the picture.",
    "Describe the image concisely.",
    "Present a compact description of the photo's key features.",
    "Give a short and clear explanation of the subsequent image.",
    "Summarize the visual content of the image.",
    "Share a concise interpretation of the image provided."
]

def tokenize(caption_file, image_dir, tokenizer, token_dump_path):
    IGNORE_INDEX = -100
    eoc = tokenizer.vocab['</s>']

    with open(caption_file, 'r') as f:
        data = json.load(f)
    
    image_id2file_name = {img['id']:img['file_name'] for img in data['images']}
    images = list(map(lambda x: image_id2file_name[x['image_id']], data['annotations']))
    images = list(map(lambda x: os.path.join(image_dir, x), images))
    print(len(images))

    captions = list(map(lambda x: x['caption'], data['annotations']))
    captions = list(map(lambda x: re.sub(' +', ' ', x.strip()), captions))
    queries = random.choices(query_template, k=len(captions))
    sample = list(map(lambda x: f'\nQ:{x[0]}<s>\nA:{x[1]}<s>', zip(queries, captions)))

    tokens = tokenizer(sample)['input_ids']
    tlen = list(map(lambda x: len(x), tokens))
    tlen = np.array(tlen)
    print(tlen.max())
    print(np.percentile(tlen, [50, 80, 90, 95, 99, 99.5]))

    pickle.dump((images, tokens), open(token_dump_path, 'wb'))

def coco_train_process():
    data_root = "/home/work/coco"
    caption_file = "annotations/captions_train2017.json"
    image_dir = "images/train2017"
    token_dump_path = "checkpoint/tokens_coco.pkl"

    caption_file = os.path.join(data_root, caption_file)
    image_dir = os.path.join(data_root, image_dir)
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    tokenize(caption_file, image_dir, tokenizer, token_dump_path)

def coco_eval_process():
    data_root = "/home/work/coco"
    caption_file = "annotations/captions_val2017.json"
    image_dir = "images/val2017"
    token_dump_path = "checkpoint/tokens_coco_eval.pkl"

    caption_file = os.path.join(data_root, caption_file)
    image_dir = os.path.join(data_root, image_dir)
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    tokenize(caption_file, image_dir, tokenizer, token_dump_path)

def LlavaPretrain_process():
    data_path = '/home/work/disk/LLaVA-data/LLAVA-Pretrain/blip_laion_cc_sbu_558k.json'
    image_dir = '/home/work/LLAVA-Pretrain/images'

    with open(data_path, 'r') as f:
        data = json.load(f)
        images = list(map(lambda x: os.path.join(image_dir, x['image']), data))
        queries = list(map(lambda x: x['conversations'][0]['value'].replace('<image>', '').strip(), data))
        captions = list(map(lambda x: x['conversations'][1]['value'], data))

    query_set = set(queries)
    # for query in query_set:
    #     print(query)

    print(len(images)) # 558128
    sample = list(map(lambda x: f'\nQ:{x[0]}<s>\nA:{x[1]}<s>', zip(queries, captions)))

    tokens = tokenizer(sample)['input_ids']
    tlen = list(map(lambda x: len(x), tokens))
    tlen = np.array(tlen)

    print(tlen.max())
    print(np.percentile(tlen, [50, 80, 90, 95, 99, 99.5]))

    x=list(zip(images, tokens))
    random.shuffle(x)
    images, tokens = zip(*x)

    token_dump_path = "checkpoint/tokens_LLAVA.pkl"
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    pickle.dump((images[:-5000], tokens[:-5000]), open(token_dump_path, 'wb'))

    token_dump_path_eval = "checkpoint/tokens_LLAVA_eval.pkl"
    token_dump_path_eval = os.path.join(cur_dir, token_dump_path_eval)
    pickle.dump((images[-5000:], tokens[-5000:]), open(token_dump_path_eval, 'wb'))

def make_sample(x):
    queries = list(filter(lambda x: x['from']=='human', x))
    queries = list(map(lambda x: x['value'].replace('<image>', '').strip(), queries))
    captions = list(filter(lambda x: x['from']=='gpt', x))
    captions = list(map(lambda x: x['value'], captions))

    sample = list(map(lambda x: f'\nQ:{x[0]}<s>\nA:{x[1]}<s>', zip(queries, captions)))
    return ''.join(sample)

def LlavaInstruct_process():
    data_path = '/home/work/disk/LLaVA-data/LLAVA-Instruct-150K/llava_v1_5_mix665k.json'
    image_dir = '/home/work/LLAVA-Instruct'

    with open(data_path, 'r') as f:
        data = json.load(f)
        data = list(filter(lambda x: 'image' in x, data))
        images = list(map(lambda x: os.path.join(image_dir, x['image']), data))
        sample = list(map(lambda x: make_sample(x['conversations']), data))

    print(len(images))

    x = list(zip(images, sample))
    x = list(filter(lambda x: os.path.exists(x[0]), x))
    images, sample = zip(*x)
    print(len(images))

    tokens = tokenizer(sample)['input_ids']
    tlen = list(map(lambda x: len(x), tokens))
    tlen = np.array(tlen)

    print(tlen.max())
    print(np.percentile(tlen, [50, 80, 90, 95, 99, 99.5]))

    x = list(zip(images, tokens))
    random.shuffle(x)
    images, tokens = zip(*x)

    token_dump_path = "checkpoint/tokens_Instruct_LLAVA.pkl"
    token_dump_path = os.path.join(cur_dir, token_dump_path)
    pickle.dump((images[:-5000], tokens[:-5000]), open(token_dump_path, 'wb'))

    token_dump_path_eval = "checkpoint/tokens_Instruct_LLAVA_eval.pkl"
    token_dump_path_eval = os.path.join(cur_dir, token_dump_path_eval)
    pickle.dump((images[-5000:], tokens[-5000:]), open(token_dump_path_eval, 'wb'))

if __name__ == "__main__":
    config = RetrieverVLConfig_medium()
    cur_dir = "/home/work/disk/vision/retriever-vl"
    tokenizer_path = "pretrain/tokenizer_v2_600G.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(cur_dir, tokenizer_path))

    # LlavaPretrain_process() # 558128 88 [30. 35. 39. 42. 48. 50.]
    # coco_train_process() # 591753 85 [30. 33. 35. 36. 41. 44.]
    # coco_eval_process() # 25014 75 [30. 33. 35. 36. 41. 44.]

    LlavaInstruct_process() # 624255 4714 [139. 419. 496. 510. 655. 759.]