import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import spacy

import boto3
import os

s3 = boto3.resource('s3')

# Specify the S3 bucket and directory that contains the checkpoint files
bucket_name = 'glyphic-ai'
directory_name = 'checkpoint-11000'

# Create the local directory if it doesn't exist
if not os.path.exists('./checkpoint-11000'):
    os.makedirs('./checkpoint-11000')

# Specify the local directory on Elastic Beanstalk where you want to save the checkpoint files
local_directory = './'


# Download all files in the S3 directory to the local directory
bucket = s3.Bucket(bucket_name)
for obj in bucket.objects.filter(Prefix=directory_name):
    if not os.path.exists(os.path.dirname(local_directory + obj.key)):
        os.makedirs(os.path.dirname(local_directory + obj.key))
        bucket.download_file(obj.key, local_directory + obj.key)


ckpt_path = './checkpoint-11000'
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, force_download=False)

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def run_model(input_string, model, tokenizer, device, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(torch.device(device))
    res = model.generate(
        input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output

def get_entities(text):
    seen = set()
    entities = []
    spacy_nlp = spacy.load('en_core_web_sm')
    for entity in spacy_nlp(text).ents:
        if entity.text not in seen:
            seen.add(entity.text)
            entities.append(entity)
    return sorted(entities, key=lambda e: e.text)


def generate_question(context, answer):
    return run_model(f"generate question: {answer} context: {context}", model, tokenizer, 'cpu', max_length=50)