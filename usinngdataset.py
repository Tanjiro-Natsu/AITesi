import tkinter as tk
from tkinter import *
import os
from tkinter.filedialog import askopenfilename
from tkinter import scrolledtext

global f


window = tk.Tk()
window.geometry("600x300")
window.title("Training Session!")
window.resizable(False, False)
window.configure(background="grey")
window.grid_columnconfigure(0, weight=1)
filechooser = tk.Button(text="File .dat",height=1, width=5,font=("Helvetica", 10))
filechooser.grid(row=0, column=0, sticky="WE", pady=10, padx=150)
text_output = tk.Text(window, wrap='word', height=5, width=10,font=("Helvetica", 15))
text_output.grid(row=1, column=0, sticky="WE",pady=10, padx=10)
Like_button = tk.Button(text="Like",height=1, width=5,font=("Helvetica", 10))#, command=nome funzione 
Like_button.grid(row=1, column=1, sticky="WE", pady=1, padx=10)
Dislike_button = tk.Button(text="Dislike",height=1, width=5,font=("Helvetica", 10))#, command=nome funzione 
Dislike_button.grid(row=1, column=2, sticky="WE", pady=10, padx=10)
Liked_button = tk.Button(text="Liked",height=1, width=5,font=("Helvetica", 10))#, command=nome funzione 
Liked_button.grid(row=2, column=0, sticky="WE", pady=1, padx=150)
Disliked_button = tk.Button(text="Disliked",height=1, width=5,font=("Helvetica", 10))#, command=nome funzione 
Disliked_button.grid(row=3, column=0, sticky="WE", pady=10, padx=150)
def filedatachooser():
  global filedatchoose
  filedatchoose=askopenfilename(filetypes=(('dat files', 'dat'),))
  filechooser.config(text=filedatchoose.split("/")[-1])
  global f
  f = open(filedatchoose, "r") 
  f.readline
  text_output.insert(tk.END,f.readline().split("::")[1])
filechooser.config(command=filedatachooser)#, command=nome funzione  
def likefunction():
  text_output.config(state=tk.NORMAL)
  text_output.delete('1.0', END)
  like=f.readline()
  text_output.insert(tk.END,like.split("::")[1])
  text_output.config(state=tk.DISABLED)
  if(os.path.isfile(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat")):
   if(os.path.isfile(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat")): 
    with open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat", "r") as input:
     with open("temp.dat", "w") as output:
        for line in input:
            if line!=like[3:]:
                output.write(line)
    os.replace('temp.dat',filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat")
   flikeread= open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat", "r")
   for line in flikeread:
     i=0
     if(line==like[3:]):
       i=1
       break
   if(i==0):
    flike = open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat", "a")
    flike.write(like.split("::")[1]+"---"+like.split("::")[2])
    flike.close()  
  else:
   flike = open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat", "a")
   flike.write(like.split("::")[1]+"---"+like.split("::")[2])
   flike.close()
def dislikefunction():
  text_output.config(state=tk.NORMAL)
  text_output.delete('1.0', END)
  dislike=f.readline()
  text_output.insert(tk.END,dislike.split("::")[1])
  text_output.config(state=tk.DISABLED)
  if(os.path.isfile(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat")):
   if(os.path.isfile(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat")): 
    with open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat", "r") as input:
     with open("temp.dat", "w") as output:
        for line in input:
            if line!=dislike[3:]:
                output.write(line)
    os.replace('temp.dat',filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat")
   fdislikeread= open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat", "r")
   y=0
   for line in fdislikeread:
     if(line==dislike.split("::")[1]+"---"+dislike.split("::")[2]):
       y=1
       break
   if(y==0):
    fdislike = open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat", "a")
    fdislike.write(dislike.split("::")[1]+"---"+dislike.split("::")[2])
    fdislike.close()  
  else:
   disflike = open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat", "a")
   disflike.write(dislike.split("::")[1]+"---"+dislike.split("::")[2])
   disflike.close()
def likedmovies():
  windowlike = tk.Tk()
  windowlike.geometry("500x500")
  windowlike.title("Liked Movies")
  windowlike.resizable(False, False)
  windowlike.configure(background="grey")
  text_outputliked = scrolledtext.ScrolledText(windowlike, wrap='word', height=500, width=500,font=("Helvetica", 15))
  text_outputliked.pack(pady=20,padx=20)
  if(os.path.isfile(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat")):
     text=""
     temp=open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"likemovies.dat", "r")
     for line in temp:
       if(text==""):
        text=line.split("---")[0]+"\n"
       else:
        text=text+line.split("---")[0]+"\n"
     text_outputliked.insert(tk.END,text)
def dislikedmovies(): 
  windowdislike = tk.Tk()
  windowdislike.geometry("500x500")
  windowdislike.title("Liked Movies")
  windowdislike.resizable(False, False)
  windowdislike.configure(background="grey")
  text_outputdisliked = scrolledtext.ScrolledText(windowdislike, wrap='word', height=500, width=500,font=("Helvetica", 15))
  text_outputdisliked.pack(pady=20,padx=20)
  if(os.path.isfile(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat")):
      text=""
      temp=open(filedatchoose.split("/")[-1].split(".dat")[0]+"-"+"dislikemovies.dat", "r")
      for line in temp:
       if(text==""):
        text=line.split("---")[0]+"\n"
       else:
        text=text+line.split("---")[0]+"\n"
      text_outputdisliked.insert(tk.END,text)
Like_button.config(command=likefunction)
Dislike_button.config(command=dislikefunction)
Liked_button.config(command=likedmovies)
Disliked_button.config(command=dislikedmovies)


if __name__ == "__main__":
    window.mainloop()
    if(f!=None):
     f.close()
    