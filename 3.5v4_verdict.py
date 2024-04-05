import os
import pandas as pd
import openai
import backoff
import re
import csv

# Setup OpenAI for Azure    
os.environ["OPENAI_API_KEY"] = ''
os.environ["TIKTOKEN_CACHE_DIR"] =""
openai.api_type = "azure"
openai.api_base = "https://gpt-4-proper.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def io_gpt(sentence, gpt_model="gpt4", temp=0, max_tokens=512, top_p=0.95):        
    system_string = """We would like to request your feedback on the performance of a AI assistant in predicting diagnosis of a histopathology image. The user gives AI assistant generated diagnosis and ground-truth diagnosis.\n Please rate the semantic matching, clinical correctness, relevance, accuracy and level of details of the AI assistant's diagnosis. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n
Please first output a single line containing only one value indicating the score for the Assistant. 
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
 
    example_messages = [{"role":"system","content":system_string}]
    
    user_message = {"role": "user", "content": sentence}
    
    # Combine the predefined examples with the new user message
    messages = example_messages + [user_message]
    
    # Make the API call with the combined message history
    response = completions_with_backoff(engine=gpt_model, temperature=temp, messages=messages, max_tokens=max_tokens, top_p=top_p, frequency_penalty=0, presence_penalty=0)
    return response
         
        
if __name__ == "__main__":
    directory = 'C:/Users/stlp/Desktop/Liinda/GPT/' 
    gpt_model= "gpt4"
    max_tokens =50 
    
    # Slice the DataFrame to get rows 3 to 20 (inclusive)
    result = []

    rows = []
    with open('GPT3.5vs4.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)    

    with open('GPT3.5vs4.csv', 'w', newline='') as csvfile:
    # Create a CSV reader object
        csvwriter = csv.writer(csvfile)        
        i= 0    
        # Iterate over each row in the CSV file
        for row in rows:
            if(i>50):                          # TO save tokens for testing
                break
            i+=1
            # Process each row as needed
            print(row[1], row[2])  # Example: Print each row

            message = "AI assistant's diagnosis:"+row[1]+"\nGround-truth diagnosis:"+row[2]

            try:
                response = io_gpt(message, gpt_model=gpt_model, max_tokens =max_tokens)
            except:
                continue    
            findings_string = response['choices'][0]['message']['content'][0]

            result.append(int(findings_string))
            csvwriter.writerow([row[0], row[1], row[2], row[3], int(findings_string)])  
        result = result[1:]
        print(result)

        print("Total Average:",sum(result)/len(result))

