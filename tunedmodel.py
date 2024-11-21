import time
import tkinter as tk
from tkinter import *
import requests
import google.generativeai as genai
import os
genai.configure(api_key="AIzaSyB3G6W8TUCuE_qAjTRK5EoDCUZJDcEu4ZU")

base_model = "models/gemini-1.5-flash-001-tuning"
training_data = [
    {"text_input": "Matrix", "output": "Truman Show,Il tredicesimo piano, eXistenZ Mulholland Drive ,Vanilla Sky ,Sinecdoche New York,Westworld"},
    {"text_input": "Notte da Leoni", "output":"Sole a catinelle,Le amiche della sposa,Ace Ventura – L'acchiappanimali,Smetto quando voglio,Santa Maradona"},
    {"text_input": "Harry Potter", "output": "Hunger Games,Il Signore degli anelli,Per Pan,Maze Runner,The Twilight Saga,Il labirinto del fauno,Maleficent,Animali fantastici e dove trovarli."},
]
operation = genai.create_tuned_model(
    display_name="esperimento1",
    source_model=base_model,
    epoch_count=20,
    batch_size=3,
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()
#print(result)


#Test

model = genai.GenerativeModel(result.name)
input_test= input('\n\n\nInserisci il titolo di un film per sapere altri film simili oppure inserisci END per terminare\n:')
while input_test!="END":
 result1 = model.generate_content(input_test)
 input_test_splitted=result1.text.split(",")
 print("\n")
 for n in input_test_splitted:
  print(n)
 input_test_splitted.clear() 
 input_test=input("\n\n\n>>>") 