import pandas as pd
import csv
fileopen=open("movies.dat",'r')
fileopen3=open("DPODatasetMovies.txt","w")
for line in fileopen:
      Categories=line.split("::")[2].split("|")
      ii=0
      k=0
      j=0
      acceptedmovies=""
      rejectedmovies=""
      fileopen2=open("movies.dat",'r')
      for line2 in fileopen2:
            if(line.split("::")[1]==line2.split("::")[1]):
                  continue
            else:
                  Categories2=line2.split("::")[2].split("|")
                  
                  
                  a_set=set(Categories)
                  b_set=set(Categories2)               
                  if(a_set & b_set):
                     if(j<5):
                        if(j==0):
                         acceptedmovies=line2.split("::")[1] 
                         j+=1
                        else:
                           acceptedmovies+="-"+line2.split("::")[1]
                           j+=1   
                  else:
                   if(k<5):
                        if(k==0):
                         rejectedmovies=line2.split("::")[1]
                         k+=1
                        else:
                           rejectedmovies+="-"+line2.split("::")[1]
                           k+=1 
                           
      fileopen3.write("Recommended film if you watched:"+line.split("::")[1]+"|"+acceptedmovies+"|"+rejectedmovies+"\n\n") 
      fileopen2.close()


fileopen.close()
fileopen3.close()      
with open("DPODatasetMovies.txt", 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split("|") for line in stripped if line)
        with open('data.csv', 'w', newline='', encoding='UTF-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('Prompt', 'Accepted', 'Rejected'))
            writer.writerows(lines)                                    
                       



