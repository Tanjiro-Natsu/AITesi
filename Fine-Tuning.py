import csv
import os
import torch
from trl import SFTTrainer
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments,Trainer
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
def formated_prompt(prompt,liked):
 return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou're a movies recommendder.....<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{liked}<|ieot_id|>\n\n"

output_model="Llama3.1-8B-Instruct-FineTunedByMarco"

while True:
 Input=input("Inserire il nome del file contenente il DataSet desiderato in formato csv\n>>")
 if(len(Input.split('.'))>1 and Input.split('.')[1]=="csv"):
  break
 else:
  os.system("cls")
path=input("Inserire il percorso in cui è presente Llama3.1-8B-Instruct \n>>")
while True:
 if(os.path.exists(path)):
  break
 else:
  os.system("cls")

  path=input("Percorso o nome file errati\n>>")
tokenizer=AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
     "facebook/opt-350m"
  )
pandas_df=pd.read_csv(Input)

data= open (Input,"r",encoding="UTF-8",newline="")
reader = csv.DictReader(data)
text_row_formatted=list()
for row in reader:
   text_row_formatted.append(formated_prompt(row["Prompt"],row["Liked"]))
 

pandas_df["text"]=text_row_formatted
pandas_df=pandas_df.drop("Disliked", axis='columns')
print(pandas_df)
pandas_df=Dataset.from_pandas(pandas_df)
print(pandas_df)
dataset=pandas_df.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
'''training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    remove_unused_columns=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pandas_df,
    eval_dataset=pandas_df,
)
trainer.train()

'''
