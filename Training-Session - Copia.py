import tkinter as tk
from tkinter import *
import requests
import google.generativeai as genai
import os
 

 
genai.configure(api_key="AIzaSyB3G6W8TUCuE_qAjTRK5EoDCUZJDcEu4ZU")
def get_gemini_response(input):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([input])
        return response.text
    except Exception as e:
        print("err in gemini ai connection",e)

window = tk.Tk()
window.geometry("1200x600")
window.title("Training Session!")
window.resizable(False, False)
window.configure(background="grey")
window.grid_columnconfigure(0, weight=1)
def print_function():
     text = text_input.get("1.0",'end-1c') 
     text=text+"\n"
     text_input.delete(1.0, tk.END)
     text_output.config(state=tk.NORMAL)
     text_output.insert(tk.END,text,"warning")
     text_output.insert(tk.END,get_gemini_response(text))
     text_output.config(state=tk.DISABLED)
 
#welcome_label = tk.Label(window,
  #                       text="Welcome! Aggiungi una parola o una frase da scaricare:",
 #                        font=("Helvetica", 15))
#welcome_label.grid(row=0, column=0, sticky="N", padx=20, pady=10)

text_output = tk.Text(window, wrap='word', height=15, width=10,font=("Helvetica", 20))
text_output.grid(row=0, column=0, sticky="WE",pady=10, padx=10)
text_output.insert(tk.END,"In cosa posso aiutarti padrone?\n")
text_output.tag_configure("warning", foreground="green")
text_output.config(state=tk.DISABLED)



text_input = tk.Text(window, wrap='word', height=2, width=10,font=("Helvetica", 20))
text_input.grid(row=1, column=0, sticky="WE",pady=10, padx=10)

Invia_button = tk.Button(text="Invia",height=1, width=10,font=("Helvetica", 20),command=print_function)#, command=nome funzione 
Invia_button.grid(row=1, column=1, sticky="WE", pady=10, padx=10)


if __name__ == "__main__":
    window.mainloop()