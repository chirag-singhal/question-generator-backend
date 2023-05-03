import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import spacy


s3 = boto3.client('s3')

# Specify the name of the S3 bucket and the checkpoint file name
bucket_name = 'glyphic-ai'
checkpoint_file_name = 'checkpoint-11000'

# Load the checkpoint file from the S3 bucket
obj = s3.get_object(Bucket=bucket_name, Key=checkpoint_file_name)
checkpoint_data = obj['Body'].read()


ckpt_path = './checkpoint/checkpoint-11000'
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_data, force_download=False)

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