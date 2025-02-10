from google.colab import drive
drive.mount("/content/drive") #permette a google collab di utilizzare file presenti in google drive


!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
!pip install --upgrade --no-cache-dir transformers
# installazione librerie necessarie per unsloth



import pandas as pd
from datasets import Dataset

df=pd.read_csv("/content/drive/MyDrive/Tesi/DataSetTest.csv")
df=Dataset.from_pandas(df)
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 
dtype = None
load_in_4bit = True 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/content/drive/MyDrive/Tesi/Llama3.2-thesis",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
#caricamento dataset e modello trainato


FastLanguageModel.for_inference(model) 
trainper=0
for i in range(len(df)):
  counter=0
  messages = [{"role": "system","content": "I need movie recommendations so pretend you are MovieGPT an ai model created by openAI to give people good movie recommendations. I'll give you a list of my favorite movies. Then give me a list of new recommended movies based on my preferences"},{"role": "user", "content": df[i]["prompt"]},]
  inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,return_tensors = "pt").to("cuda")
  generated_ids = model.generate(inputs, do_sample=False, max_new_tokens=1024)
  stringaoutput=tokenizer.batch_decode(generated_ids[:, inputs.shape[1]:], skip_special_tokens=True)[0]
  films=df[i]["chosen"].split("-")
  for k in range (len(films)):
    if(films[k] in stringaoutput):
      counter+=1
  per=100 * float(counter)/float(len(films))
  trainper+=per
  print(str(i+1)+":"+str(per))
if(trainper!=0):
 print(100 * float(trainper)/float(len(len(df))))
#test