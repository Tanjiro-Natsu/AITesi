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



alpaca_prompt = """Below is an instruction that ask for a list of movies, paired with an input that provides a list of movies that you liked . Complete the response with a list based on the file ratings.dat.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token 
def format_prompt(sample):
    instruction = "You're a movies recommendder. You will be given a list of liked film . You must generate a list with movies based on the list that i gave you."
    input       = sample["prompt"]
    accepted    = sample["chosen"]
    rejected    = sample["rejected"]
    sample["prompt"]   = alpaca_prompt.format(instruction, input, "")
    sample["chosen"]   = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    return sample
pass
import pandas as pd
from datasets import Dataset
df=pd.read_csv("/content/drive/MyDrive/DataSetTraining.csv")
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
        num_train_epochs = 1,
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



model.save_pretrained("Llama3.2-thesis")
tokenizer.save_pretrained("Llama3.2-thesis")
#salvataggio modello localmente









