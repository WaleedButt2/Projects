import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
class PersonalDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
training_args = TrainingArguments(output_dir='./results', num_train_epochs=5, logging_steps=50, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=15, per_device_eval_batch_size=15, warmup_steps=50,
                                 weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./ds_config.json')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
model.resize_token_embeddings(len(tokenizer))
descriptions = pd.read_csv('../data.text')['text']
max_length = max([len(tokenizer.encode(description)) for description in descriptions])
print("Max length: {}".format(max_length))
print(torch.cuda.memory_allocated())
dataset = PersonalDataset(descriptions, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
Trainer(model=model, args=training_args, train_dataset=train_dataset,
       eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])}).train()
input_text = "<|startoftext|> Male Muqeeet, one of your closest freinds in the university. He had decided to help you set up a girl and went out with you and her to help smooth it along. But as it turns out he was sitting by her and talking/laughing while excluding you. You are sitting in KFC with muqeet and the girl nameed AX2. Both muqeet and AX2 are sitting together. You: I think we should switch seats right muqeet? Muqeet:"
encoded_input = tokenizer(input_text, return_tensors="pt").input_ids
sample_output=model.generate(encoded_input, 
                do_sample=True, 
                top_k=50,
                max_length=300, 
                top_p=0.5, 
                temperature=0.7,
                num_return_sequences=1)
output_text = tokenizer.decode(sample_output[0],skip_special_tokens=True)
print(output_text)
model.save_pretrained("./Flaskapp/uo")