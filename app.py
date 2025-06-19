import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Project Ideas Dashboard", layout="wide")

# Load project ideas
@st.cache_data
def load_project_ideas():
    csv_path = "C:/Users/Simphiwe Mbatha/Desktop/Project2/project_ideas.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # Sample data
    data = [
        {"Title": "Fresh and Stale Classification", "Description": "Classify fruits/vegetables as fresh or stale using CNNs", "Category": "Machine Learning", "Status": "In Progress", "Page": "main8"},
        {"Title": "Household Electricity Usage Calculator", "Description": "Analyze and optimize household electricity usage", "Category": "Data Analysis", "Status": "Completed", "Page": "main5"}
    ]
    return pd.DataFrame(data)

# Main dashboard
def main():
    st.title("Project Ideas Dashboard")
    st.write("Click a project to view details and interact with the app.")

    df = load_project_ideas()
    
    if df.empty:
        st.error("No project ideas found. Add data to 'project_ideas.csv'.")
        return

    st.subheader("Project Ideas")
    cols = st.columns(2)
    for idx, row in df.iterrows():
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div style='border: 1px solid #e6e6e6; border-radius: 5px; padding: 10px; margin: 10px;'>
                    <h3>{row['Title']}</h3>
                    <p><b>Category:</b> {row['Category']}</p>
                    <p><b>Status:</b> {row['Status']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"View {row['Title']}", key=f"view_{idx}"):
                st.session_state.page = row['Page']
                st.experimental_rerun()

if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    main()