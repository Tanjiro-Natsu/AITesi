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
fileopentraining.write("prompt/Liked/Disliked\n")
fileopentest.write("prompt/Liked/Disliked\n")
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
     # listlentght=len(liked)
     # promptid=(int)(listlentght*0.8)
     # fileopenmovies=open(filetopenmovies,'r')
     # set1=set(liked[:promptid])
     # set2=set(liked[promptid:])
     # set3=set(disliked)
     # for movies in fileopenmovies:
     #   if(movies.split("::")[0] in set1):
     #     if(prompt==""):
     #       prompt=movies.split("::")[1]
     #     else:
     #       prompt+="-"+movies.split("::")[1]  
     #   if(movies.split("::")[0] in set2):
     #     if(Liked==""):
     #       Liked=movies.split("::")[1]
     #     else:
     #       Liked+="-"+movies.split("::")[1] 
     #   if(movies.split("::")[0] in set3):
     #     if(Disliked==""):
     #       Disliked=movies.split("::")[1]
     #     else:
     #       Disliked+="-"+movies.split("::")[1]
     # fileopenmovies.close()           
      if(checkid<=p1): 
       fileopentraining.write(str(checkid)+"::"+str(liked[:promptid])+"/"+str(liked[promptid:])+"/"+str(disliked)+"\n\n")
      else:
       fileopentest.write(str(checkid)+"::"+str(liked[:promptid])+"/"+str(liked[promptid:])+"/"+str(disliked)+"\n\n")
      liked.clear()
      disliked.clear()
      if((int)(line.split("::")[2])>3):
        liked.append(line.split("::")[1])
      else:
        disliked.append(line.split("::")[1])
      checkid=(int)(line.split("::")[0].replace(" ",""))
       
fileopentest.write(str(checkid)+"::"+str(liked[:promptid])+"/"+str(liked[promptid:])+"/"+str(disliked)+"\n\n")
fileopentraining.close()
fileopentest.close()  


fileopen=open(filetopentraining,"r")
stringa=""
for row in fileopen:
        if(row==""):
         with open('dataratings.csv', 'w', newline='', encoding='UTF-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(stringa)
            stringa=""
        else:
          if(stringa==""):
            stringa=row
          else:
            stringa+=row
fileopen.close()
   

