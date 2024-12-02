from transformers import AutoTokenizer
import os
path=input("Inserire il percorso in cui è presente Llama3.1-8B-Instruct \n>>")
tokenizer=AutoTokenizer.from_pretrained(path)#"C:\\Users\\marco\\Desktop\\AITesi\\C-\\Users\\marco\\.llama\\checkpoints\\Llama3.1-8B-Instruct")#
k=0
loop=0
while loop==0:
 filetopen=input("Inserire il nome del file che deve essere tokenizato con TikToken\n>>")
 try:
  fileopen=open(filetopen,'r')
  loop=1
 except:
     loop=0
     os.system("cls")
     print("File inserito non trovato\n")
loop=0
#os.system("cls")
filetokenized=input("Inserire il nome del file tokenizato con TikToken\n>>")
filekenized=open(filetokenized,"w")
for line in fileopen:
  if(line=="\n"):
    filekenized.write("\n")
    continue
  string="[ "
  for word in line.split(" "):
    if(word=="/"):
        string+="] / [ "  
    else:
     
     if(len(tokenizer(word)['input_ids'])>1):
      for i in range(len(tokenizer(word)['input_ids'])): 
        string+=str(tokenizer(word)['input_ids'][i])+" "
     else:
       temporary=str(tokenizer(word)['input_ids']).replace("[","")
       temporary=temporary.replace("]","")
       string+=temporary+" "  
  filekenized.write(string +"]"+"\n")
filekenized.close()
fileopen.close()
 
  

