import streamlit as st
import json
import gspread
from google.oauth2.service_account import Credentials
import os

# Convert secrets group into a dict
sa_info = dict(st.secrets["gcp_service_account"])

# gspread requires credentials as a dict; create Credentials
creds = Credentials.from_service_account_info(sa_info, scopes=[
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"  # optional, only if you need Drive access
])

gc = gspread.authorize(creds)

st.set_page_config("DPO Dataset Generation", layout="wide")

# Initialize session state


# Load datasets

script_dir = os.path.dirname(__file__)  # folder containing interface.py
json_path = os.path.join(script_dir, "Choices_to_compare.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sheet_id = "1K-vuVHxuEniSyWv4y0lHv5OZ-ohmpNSDDrKHh1aN2KE"
sh = gc.open_by_key(sheet_id)
ws = sh.sheet1 

Markeddata = ws.get_all_records()

nums = ws.col_values(1)
if "num_row" not in st.session_state:
    nums = ws.col_values(1)
    if len(nums) > 1:
        last_number = int(nums[-1])
    else:
        last_number = 0  
    st.session_state.num_row = last_number

st.title("DPO Dataset Generation for M37")
st.write("Number of ungraded choices: ", len(data)- len(Markeddata))
st.write("Number marked: ", len(Markeddata))

def process_row(chosen, rejected):

    ws.append_row([st.session_state.num_row, chosen, rejected, ])
    
    st.session_state.num_row += 1

    st.rerun()
    show_row()


def show_row():
    if st.session_state.num_row >= len(data):
        st.success("All rows processed!")
        return

    row = data[st.session_state.num_row]
    one = row["one"]
    two = row["two"]
    left, right = st.columns(2)

    with left:
        st.write(one)
        if st.button("Choose one"):
            process_row(one, two)
    with right:
        st.write(two)
        if st.button("Choose two"):
            process_row(two, one)

show_row()
