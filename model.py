from transformers import pipeline

# Defining directly the task we want to implement. 
pipe = pipeline(task="sentiment-analysis")

pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def generate_response(prompt:str):
   response = pipe("This is a great tutorial!")
   label = response[0]["label"]
   score = response[0]["score"]
   return f"The '{prompt}' input is {label} with a score of {score}"

print(generate_response("This tutorial is great!"))