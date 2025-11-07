import streamlit as st
from openai import OpenAI
import json
import os
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()

def write_to_db(chosen, rejected):
    with open("DPO-Created-Dataset", "r+") as f:
        try:
            previous_data = json.load(f)
        except Exception as e:
            previous_data = []
        new_row = {"chosen": chosen, "rejected": rejected}
        previous_data.append(new_row)
        f.seek(0)
        json.dump(previous_data, f, indent=2)

st.set_page_config(page_title="Dpo dataset generator", layout="wide")

st.title("Dpo dataset generator")

System_prompt = st.text_area("Enter the system prompt")

User_input = st.text_area("Enter the user prompt")

response1 = "Human:" + User_input +"Assistant:" + client.responses.create(
    model = "gpt-4",
    instructions=System_prompt,
    input=User_input
)

response2 = "Human:" + User_input +"Assistant:" + client.responses.create(
    model = "gpt-4",
    instructions=System_prompt,
    input=User_input
)

col1, col2 = st.columns(2)

with col1:
    st.write("Response 1: ", response1)
    if st.button("Choose choice 1."):
        write_to_db(chosen=response1, rejected=response2)

with col2:
    st.write("Response 2: ", response2)
    if st.button("Choose choice 2."):
        write_to_db(chosen=response2, rejected=response1)



