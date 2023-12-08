from openai import OpenAI
import json
import pandas as pd

client = OpenAI(api_key='')



def chatgpt_translate(row):
    
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-16k",
        messages=[{
            "role":"system",
            "content":""
        },
        {
            "role":"user",
            "content":f"Translate from English to Spansh:{row['English']}"
        }],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

        




df = pd.read_table('./comparative_samples.txt',header=None,names=['English'])


df.loc[:,'Spanish'] = df.apply(chatgpt_translate,axis=1)


df.to_csv('./comparative_samples_chatgpt_translated.csv',index=False)
