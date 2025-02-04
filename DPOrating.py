import pandas as pd
import csv
import os
loop=0
while loop==0:
 filetopen=input("Inserire il nome del file rating che si desiderà trasformare in dataset\n>>")
 try:
  fileopen=open(filetopen,'r')
  loop=1
 except:
     loop=0
     os.system("cls")
     print("File inserito non trovato\n")
loop=0
os.system("cls")
filetopenmovies=""
while loop==0:
 filetopenmovies=input("Inserire il nome del file che contiene la lista di tutti i film\n>>")
 try:
  fileopenmovies=open(filetopenmovies,'r')
  loop=1
 except:
     loop=0
     os.system("cls")
     print("File inserito non trovato\n")       
counter=0
pointer=0
for line in fileopen:
      UserID=line.split("::")[0]
      if(pointer!=UserID):
            pointer=UserID
            counter+=1
fileopenmovies.close()   
fileopen.close()
p1=(int)(counter*0.8)
p2=(int)(counter*0.2)
os.system("cls")
fileopen=open(filetopen,'r')

q=0
filetopentest=""
filetopentraining=""
while q==0:
  filetopentest=input("Inserire il nome del file contenente il dataset da creare per il test\n>>")
  if ( len(filetopentest.split('.'))==2):
   if (filetopentest.split('.')[1]=="dat"):
     q=1
   else:
     q=0   
     os.system("cls")
     print("Nome file errato,inserire estenzione .dat alla fine del file\n")
  else:
     q=0   
     os.system("cls")
     print("Nome file errato,inserire estenzione .dat alla fine del file\n")

w=0
while w==0:
  filetopentraining=input("Inserire il nome del file contenente il dataset da creare per il training\n>>")
  if ( len(filetopentraining.split('.'))==2):
   if (filetopentraining.split('.')[1]=="dat" ):
     w=1
   else:
     w=0   
     
     os.system("cls")
     print("Nome file errato,inserire estenzione .dat alla fine del file\n")
  else:
     w=0   
     os.system("cls")
     print("Nome file errato,inserire estenzione .dat alla fine del file\n")
    

fileopentest=open(filetopentest,'w')
fileopentraining=open(filetopentraining,'w')
checkid=1
promptid=0
liked=list()
disliked=list()
prompt=""
Liked=""
Disliked=""
for line in fileopen:
    
    if(line.split("::")[0].replace(" ","")==str(checkid)):
      if((int)(line.split("::")[2])>3):
        liked.append(line.split("::")[1])
      else:
        disliked.append(line.split("::")[1])    
    else:
      listlentght=len(liked)
      promptid=(int)(listlentght*0.8)
      listlentght=len(liked)
      promptid=(int)(listlentght*0.8)
      fileopenmovies=open(filetopenmovies,'r')
      set1=set(liked[:promptid])
      set2=set(liked[promptid:])
      set3=set(disliked)
      for movies in fileopenmovies:
        if(movies.split("::")[0] in set1):
          if(prompt==""):
            prompt=movies.split("::")[1]
          else:
            prompt+=" - "+movies.split("::")[1]  
        if(movies.split("::")[0] in set2):
          if(Liked==""):
            Liked=movies.split("::")[1]
          else:
            Liked+=" - "+movies.split("::")[1] 
        if(movies.split("::")[0] in set3):
          if(Disliked==""):
            Disliked=movies.split("::")[1]
          else:
            Disliked+=" - "+movies.split("::")[1]
      fileopenmovies.close()           
      if(checkid<=p1): 
       fileopentraining.write(prompt+" / "+Liked+" / "+Disliked+"\n\n")#fileopentest.write(str(checkid)+"::"+str(liked[:promptid])+"/"+str(liked[promptid:])+"/"+str(disliked)+"\n\n")
       prompt=""
       Liked=""
       Disliked=""      
      else:
       fileopentest.write(prompt+" / "+Liked+" / "+Disliked+"\n\n")#fileopentest.write(str(checkid)+"::"+str(liked[:promptid])+"/"+str(liked[promptid:])+"/"+str(disliked)+"\n\n")
       prompt=""
       Liked=""
       Disliked=""      
      liked.clear()
      disliked.clear()
      if((int)(line.split("::")[2])>3):
        liked.append(line.split("::")[1])
      else:
        disliked.append(line.split("::")[1])
      checkid=(int)(line.split("::")[0].replace(" ",""))
   
        
fileopentest.write(prompt+" / "+Liked+" / "+Disliked+"\n\n")#fileopentest.write(str(checkid)+"::"+str(liked[:promptid])+"/"+str(liked[promptid:])+"/"+str(disliked)+"\n\n")
fileopentraining.close()
fileopentest.close()  

import csv 
rows=list()
k=0
with open(filetopentest,"r") as in_file:
 for line in in_file:
  lines=line.split("/")
  if(len(lines)==3):
   PromptEmptyStrings=""
   for line1 in range(len(lines[0].split("-"))):
       singlefilm=lines[0].split("-")[line1]
       if(line1==len(lines[0].split("-"))-1):
        PromptEmptyStrings=PromptEmptyStrings+singlefilm.split("(")[0]
       else:  
        PromptEmptyStrings=PromptEmptyStrings+singlefilm.split("(")[0]+"-"

   LikedEmptyStrings=""
   for line1 in range(len(lines[1].split("-"))):
       singlefilm=lines[1].split("-")[line1]
       if(line1==len(lines[1].split("-"))-1):
        LikedEmptyStrings=LikedEmptyStrings+singlefilm.split("(")[0]
       else:  
        LikedEmptyStrings=LikedEmptyStrings+singlefilm.split("(")[0]+"-"

   DislikedEmptyStrings=""
   for line1 in range(len(lines[2].split("-"))):
       singlefilm=lines[2].split("-")[line1]
       if(line1==len(lines[2].split("-"))-1):
        DislikedEmptyStrings=DislikedEmptyStrings+singlefilm.split("(")[0]
       else:  
        DislikedEmptyStrings=DislikedEmptyStrings+singlefilm.split("(")[0]+"-"

   lines[0]=PromptEmptyStrings
   lines[1]=LikedEmptyStrings
   lines[2]=DislikedEmptyStrings
   rows.append({"prompt":lines[0],"chosen":lines[1],"rejected":lines[2]}) 
  else:
   continue
filetest=filetopentest.split(".")[0]+".csv"
with open (filetest,"w",encoding="UTF-8",newline="") as out_file:
    writer=csv.DictWriter(out_file,fieldnames=["prompt","chosen","rejected"])
    writer.writeheader()
    writer.writerows(rows)
rows1=list()
filetraining=filetopentraining.split(".")[0]+".csv"
with open(filetopentraining,"r") as in_file:
 for line in in_file:
  lines=line.split("/")
  if(len(lines)==3):
   PromptEmptyStrings=""
   for line1 in range(len(lines[0].split("-"))):
       singlefilm=lines[0].split("-")[line1]
       if(line1==len(lines[0].split("-"))-1):
        PromptEmptyStrings=PromptEmptyStrings+singlefilm.split("(")[0]
       else:  
        PromptEmptyStrings=PromptEmptyStrings+singlefilm.split("(")[0]+"-"

   LikedEmptyStrings=""
   for line1 in range(len(lines[1].split("-"))):
       singlefilm=lines[1].split("-")[line1]
       if(line1==len(lines[1].split("-"))-1):
        LikedEmptyStrings=LikedEmptyStrings+singlefilm.split("(")[0]
       else:  
        LikedEmptyStrings=LikedEmptyStrings+singlefilm.split("(")[0]+"-"

   DislikedEmptyStrings=""
   for line1 in range(len(lines[2].split("-"))):
       singlefilm=lines[2].split("-")[line1]
       if(line1==len(lines[2].split("-"))-1):
        DislikedEmptyStrings=DislikedEmptyStrings+singlefilm.split("(")[0]
       else:  
        DislikedEmptyStrings=DislikedEmptyStrings+singlefilm.split("(")[0]+"-"

   lines[0]=PromptEmptyStrings
   lines[1]=LikedEmptyStrings
   lines[2]=DislikedEmptyStrings
   rows1.append({"prompt":lines[0],"chosen":lines[1],"rejected":lines[2]}) 
  else:
   continue 
with open (filetraining,"w",encoding="UTF-8",newline="") as out_file:
    writer=csv.DictWriter(out_file,fieldnames=["prompt","chosen","rejected"])
    writer.writeheader()
    writer.writerows(rows1)