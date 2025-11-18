
import streamlit as st
import json
import gspread
from google.oauth2.service_account import Credentials
import os
from openai import OpenAI

# Convert secrets group into a dict
sa_info = dict(st.secrets["gcp_service_account"])
openaiApiKey = st.secrets["openaiApiKey"]
os.environ["OPENAI_API_KEY"] = openaiApiKey

client = OpenAI()

creds = Credentials.from_service_account_info(
    sa_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive", 
    ],
)

gc = gspread.authorize(creds)

st.set_page_config("Custom DPO Dataset Generation", layout="wide")

if "num_row" not in st.session_state:
    st.session_state["num_row"] = 0

script_dir = os.path.dirname(__file__)


def generateResponse(ws, numExamples, model, systemPrompt, prompt):
    for i in range(numExamples):
        with st.spinner(f"Generating example {i+1}"):
            resp1 = client.responses.create(
                model=model,
                input=prompt,
                instructions=systemPrompt if systemPrompt else None,
            )
            resp2 = client.responses.create(
                model=model,
                input=prompt,
                instructions=systemPrompt if systemPrompt else None,
            )

            text1 = resp1.output_text
            text2 = resp2.output_text

            ws.append_row([i + 1, prompt, systemPrompt, text1, text2])


def process_row(ws_marked,num_row, chosen, rejected):

    ws_marked.append_row([num_row, chosen, rejected])
    st.session_state.num_row += 1
    st.rerun()


def show_row(ws_marked, data, num_row):

    if num_row >= len(data):
        st.success("All rows have been marked ✅")
        return

    row = data[num_row]

    _, prompt_text, sys_prompt, one, two = row[0], row[1], row[2], row[3], row[4]

    st.write(f"**Prompt:** {prompt_text}")

    left, right = st.columns(2)

    with left:
        st.subheader("Option 1")
        st.write(one)
        if st.button("Choose Option 1", key=f"choose_one_{st.session_state.num_row}"):
            process_row(ws_marked,num_row, one, two)

    with right:
        st.subheader("Option 2")
        st.write(two)
        if st.button("Choose Option 2", key=f"choose_two_{st.session_state.num_row}"):
            process_row(ws_marked,num_row, two, one)


st.title("DPO Dataset Generation for M37")

sheet_id = st.text_input("Sheet id to store dataset:", key="sheet_id_create")

createDataTab, markDataTab = st.tabs(["Create data", "Mark data"])

with createDataTab:
    

    if sheet_id:
        sh_create = gc.open_by_key(sheet_id)
        unmarkedData = sh_create.sheet1  # where pairs will be stored

        model = st.selectbox("Choose a model:", ["gpt-5", "gpt-5.1", "gpt-5.1-mini", "o3-mini", "o4-mini","gpt-4.1"])
        systemPrompt = st.text_area("System prompt")
        prompt = st.text_area("Prompt")

        numGenerate = st.number_input(
            "How many examples to generate?",
            min_value=1,
            step=1,
            value=1,
        )

        if st.button("Generate examples"):
            generateResponse(
                ws=unmarkedData,
                numExamples=int(numGenerate),
                model=model,
                systemPrompt=systemPrompt,
                prompt=prompt,
            )
            st.success("Examples generated and written to Sheet1 ✅")

with markDataTab:
    if sheet_id:
        sh_mark = gc.open_by_key(sheet_id)
        unmarkedData_mark = sh_mark.sheet1  # source pairs
        markedData = sh_mark.worksheet("Sheet2")       # DPO dataset (chosen, rejected)

        # Load all unmarked rows
        data = unmarkedData_mark.get_all_values()

        st.session_state.num_row = len(markedData.get_all_values())-1
        print(data)

        show_row(markedData, data, st.session_state.num_row)
