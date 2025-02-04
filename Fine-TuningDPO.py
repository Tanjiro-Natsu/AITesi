from google.colab import drive
drive.mount("/content/drive") #permette a google collab di utilizzare file presenti in google drive



!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
!pip install --upgrade --no-cache-dir transformers
# installazione librerie necessarie per il fine-tuning



from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 
dtype = None
load_in_4bit = True 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # utilizzare in caso di gated model
)
#caricamento modello e tokenizer



model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none", 
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)
#Applica PEFT (Parameter-Efficient Fine-Tuning).Configuraziuone  del modelo per il fine-tuning usando LoRA (Low-Rank Adaptation) specificando parametri come: rank, target modules, dropout, e gradient checkpointing tper ottimizzare la memoria e l'efficenza del training.



instruction = "I need movie recommendations so pretend you are MovieGPT an ai model created by openAI to give people good movie recommendations. I'll give you a list of my favorite movies. Then give me a list of new recommended movies based on my preferences"
def create_prompt( user_message: str,chosen: str) -> str:
    prompt = f'''
     "role": "system", "content": {instruction },
               "role": "user", "content": {user_message},
               "role": "assistant", "content": {chosen}
    '''
    return prompt

EOS_TOKEN = tokenizer.eos_token 


def format_prompt(sample):
    input       =sample["prompt"]
    accepted    = sample["chosen"]
    rejected    = sample["rejected"]
    
    sample["chosen"]   = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    sample["prompt"]   = create_prompt(input,accepted)
    return sample
pass

import pandas as pd
from datasets import Dataset

df=pd.read_csv("/content/drive/MyDrive/Tesi/DataSetTraining.csv")
df=Dataset.from_pandas(df)
df = df.map(format_prompt,)
#Preparazione DataSet




from unsloth import PatchDPOTrainer
PatchDPOTrainer()
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
    beta = 0.1,
    train_dataset = df,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
#Configurazione DPOTrainer



dpo_trainer.train()
#Inizio allenamento



model.save_pretrained("/content/drive/MyDrive/Tesi/Llama3.2-thesis") 
tokenizer.save_pretrained("/content/drive/MyDrive/Tesi/Llama3.2-thesis")
#salvataggio modello localmente



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/content/drive/MyDrive/Tesi/Llama3.2-thesis",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) 
messages = [

    {

        "role": "system",

        "content": "I need movie recommendations so pretend you are MovieGPT an ai model created by openAI to give people good movie recommendations. I'll give you a list of my favorite movies. Then give me a list of new recommended movies based on my preferences",
    

    },

    {"role": "user", "content": "Toy Story - Pocahontas - Apollo 13 - Schindler's List - Aladdin"
 },

]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,return_tensors = "pt").to("cuda")

generated_ids = model.generate(inputs, do_sample=True, max_new_tokens=1024)

print(tokenizer.batch_decode(generated_ids[:, inputs.shape[1]:], skip_special_tokens=True)[0])





