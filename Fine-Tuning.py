import csv
import os
import torch
from trl import DPOTrainer,DPOConfig
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments,Trainer
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from huggingface_hub import login


def formated_prompt(prompt,liked):
 return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou're a movies recommendder.....<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{liked}<|ieot_id|>\n\n"


output_model="Llama3.1-8B-Instruct-FineTunedByMarco"
tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
while True:
 Input=input("Inserire il nome del file contenente il DataSet desiderato in formato csv\n>>")
 if(len(Input.split('.'))>1 and Input.split('.')[1]=="csv"):
  break
 else:
  os.system("cls")
Input="DataSetTraining.csv"
pandas_df=pd.read_csv(Input)

data= open (Input,"r",encoding="UTF-8",newline="")
reader = csv.DictReader(data)
text_row_formatted=list()
for row in reader:
   text_row_formatted.append(tokenizer(formated_prompt(row["Prompt"],row["Liked"]))['input_ids'])
 

pandas_df["text"]=text_row_formatted
pandas_df=pandas_df.drop("Disliked", axis='columns')
pandas_df=Dataset.from_pandas(pandas_df)
#dataset=pandas_df.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
  )
  
training_args = DPOConfig(output_dir=output_model, logging_steps=2)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=pandas_df)
trainer.train()