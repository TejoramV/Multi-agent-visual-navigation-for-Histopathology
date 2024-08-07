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
    system_string = """Assuming you are a histopathologist, given clues create a diagnosis from it."""
    user1_string = """clues: [1. "The heart follows the classical pattern of inner lining, middle lining, outer lining", 2. "The heart, even though it's the largest blood vessel, once again, would boil down to a single endothelial cell lining along its endocardial surface", 3. "99.9% of the weight of the heart is the muscle itself, or the myocardium, which is classical cardiac muscle", 4. "The outer aspect, which would correspond to the adventitia if this was a blood vessel. But because we are in the heart, it is called the pericardium, or more specifically, the visceral pericardium", 5. "All of the major arteries of the heart travel in pericardial fat"]"""
    assistant1_string = """diagnosis: ["Normal Histology of the Heart"]"""
    user2_string ="""clues: [1. Presence of a papule on acral skin, 2. Proliferation of spindle cells in the dermis, 3. Presence of Schwann cells arranged into nerve bundles, 4. Dense collagen, 5. Presence of Meissner's corpuscles, 6. Location on the distal fifth digit, 7. Patient is a young child, 8. Similar bump on the other hand, 9. Large nerve bundles feeding into the papule]"""
    assistant2_string ="""diagnosis: ["Accessory Digit"]"""
    user3_string = """clues: [1. "benign conjunctal nevus right here", 2. "a melanoma outside here", 3. "at least one or probably two conjunctal nevus here", 4. "melanoma inside here", 5. "at least one benign nevus and another melanoma inside too, rising in the benign nevus"]"""
    assistant3_string = """diagnosis: ["Benign Conjunctal Nevus", "Melanoma"]""" 
    example_messages = [{"role":"system","content":system_string},
                        {"role":"user","content":user1_string},
                        {"role":"assistant","content":assistant1_string},
                        {"role":"user","content":user2_string},
                        {"role":"assistant","content":assistant2_string},                        
                        {"role":"user","content":user3_string},
                        {"role":"assistant","content":assistant3_string}
                        ]
    
    user_message = {"role": "user", "content": sentence}
    
    # Combine the predefined examples with the new user message
    messages = example_messages + [user_message]
    
    # Make the API call with the combined message history
    response = completions_with_backoff(engine=gpt_model, temperature=temp, messages=messages, max_tokens=max_tokens, top_p=top_p, frequency_penalty=0, presence_penalty=0)
    return response
         
        
if __name__ == "__main__":
    data = pd.read_parquet('C:/Users/stlp/Desktop/Liinda/diagnosis_and_clues.parquet')
    directory = 'C:/Users/stlp/Desktop/Liinda/GPT/' 
    gpt_model= "gpt35"
    max_tokens =50 
    message ="""clues: [1. Presence of scattered dark firm areas on multiple lobes consistent with locally extensive areas of bronchopneumonia, 2. Hyperkeratosis of the foot pad and nose, 3. Conjunctivitis and enamel hypoplasia, 4. Microscopic examination showing approximately 95% of the lung is diffusely hemorrhagic with scattered areas of necrosis, 5. Airways plugged with cellular exudate and the lining epithelium is hyperplastic, 6. Alveolar architecture replaced by marked inflammation, hemorrhage, and fibrin with edema, 7. Presence of abundant degenerate and non-degenerate neutrophils, macrophages, fibrin deposits, and abundant hemorrhage, 8. Less affected alveoli are lined by type 2 pneumocyte hyperplasia, 9. Pneumocytes frequently fuse together and form multinucleated viral syncytia containing up to 10 nuclei, 10. Presence of eosinophilic intracytoplasmic and intranuclear viral inclusion bodies, 11. Presence of viral syncytia along a combination of intracytoplasmic and intranuclear viral inclusions is consistent with canine distemper virus.]"""
    
    # Slice the DataFrame to get rows 3 to 20 (inclusive)
    subset_data = data.iloc[10:310]
    result = []

        
    # Iterate through each row in the subset
    for index, row in subset_data.iterrows():
        pattern = r'clues:'
        match = re.search(pattern, row['content'])
        if match:
            index_location = match.start()
            # print(row['content'][index_location+7:-1]) #clues without "clues:"
            message = row['content'][index_location:-1] #clues    
            try:
                response = io_gpt(message, gpt_model=gpt_model, max_tokens =max_tokens)
            except:
                continue    
            findings_string = response['choices'][0]['message']['content']

            print('Video ID:', row['video_id'])
            print(row['content'][12:index_location-2])# GPT 4 diagnosis  
            print(findings_string[11:]) # GPT 3.5 diagnosis

            result.append([row['video_id'], findings_string[11:], row['content'][12:index_location-2],row['content'][index_location+7:-1]])
        else:
            continue
    csv_file_path = "GPT3.5vs4.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow(["video_id","gpt_3.5_diagnosis","gpt_4_diagnosis","clues"])
        # Append the data to the CSV file
        for row in result:
            writer.writerow(row)
    print("Data appended successfully.")
