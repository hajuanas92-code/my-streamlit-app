text = [
  {
     "user_query": "Is there a specific dress code for attending the company annual general meeting?",
     "matched_category": "HR_Policy",
     "system_response": "I could not find an exact match for this in our FAQ database. Please contact the HR department at hr@company.com for specific dress code guidelines."
   },
   {
     "user_query": "What is the return policy for items purchased during the Black Friday sale?",
     "matched_category": "Returns_Policy",
     "system_response": "For holiday sales return policies, please refer to our main Terms of Service page.And then you will recieve a call from our team"
   },
   {
     "user_query": "Can I integrate your software with my external CRM tool?",
     "matched_category": "Integrations",
     "system_response": "There are no matching FAQs found for this query. If you need assistance with custom CRM integrations, please submit a ticket to our technical support team."
   },
   {
     "user_query": "What are the exact operating hours for your store branch in Mumbai?",
     "matched_category": "Store_Hours",
     "system_response": "I could not locate an answer to this in our current FAQs. Please visit our website's [Store Locator](https://example.com) page to check exact operating hours."
   }
 ]
import streamlit as st
import pandas as pd
import json
import chromadb
from transformers import pipeline

st.title("Welcome to ChatBot Arena")
df = pd.read_json(json.dumps(text))
client = chromadb.Client()

bot = pipeline('text-generation',model='Qwen/Qwen2.5-0.5B-Instruct')

try:
  client.delete_collection(name='my_app')
except Exception:
  pass
collection = client.get_or_create_collection(name='my_app')
chunks = [text.strip() for text in df['system_response']]
collection.add(documents=chunks,
           ids = [f"id{i}" for i in range(len(chunks))])
user = st.text_input("Ask your question")
result = collection.query(query_texts=[user],n_results=1)
retrieve = " ".join(result["documents"][0])
prompt = f"""You should answer the questions according to the given context ONLY if you don't found any related context say 'i don't know the perfect answer'.
Don't generate any answer about your own UNDERSTOOD.
Context : {retrieve}
Question : {user}
Answer:"""
result = bot(prompt,max_new_tokens=50,return_full_text=False)
st.write("Result: ",result[0]['generated_text'])

