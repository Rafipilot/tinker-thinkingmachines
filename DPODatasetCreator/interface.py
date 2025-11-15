import streamlit as st
import json

st.set_page_config("DPO Dataset Generation", layout="wide")

# Initialize session state
if "num_row" not in st.session_state:
    st.session_state.num_row = 0

# Load datasets
with open("Choices_to_compare.json", "r", encoding="utf-8") as f:
    data = json.load(f)

try:
    with open("CreatedDatset.json", "r", encoding="utf-8") as f:
        Markeddata = json.load(f)
except:
    Markeddata = []

st.title("DPO Dataset Generation for alek :)")
st.write("Number of ungraded choices: ", len(data))
st.write("Number marked: ", len(Markeddata))

def process_row(chosen, rejected):
    # Save choice
    Markeddata.append({"chosen": chosen, "rejected": rejected})
    with open("CreatedDatset.json", "w", encoding="utf-8") as f:
        json.dump(Markeddata, f, ensure_ascii=False, indent=2)
    
    # Move to next row
    st.session_state.num_row += 1

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
