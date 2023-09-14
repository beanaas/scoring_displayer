import streamlit as st
import json
import pandas as pd
# Streamlit App Title
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # You can also set the layout if needed
    initial_sidebar_state="expanded",  # You can choose "expanded" or "collapsed"
    
)

# Upload JSON File
uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read JSON data
    data = json.load(uploaded_file)

    # Convert the data to a DataFrame (optional)
    df = pd.json_normalize(data)

    # Display the DataFrame (optional)
    st.write("### Data Preview")
    df_copy = df.copy()
    columns_to_remove = ['signal1','signal2', 'signal3', 'beforeCross1', 'beforeCross2', 'beforeCross3']

    # Remove the specified columns from the copy
    df_copy.drop(columns=columns_to_remove, inplace=True)
    st.table(df_copy)
    

    # Interactive Chart 1 - Scatter Plot
    st.write("### Interactive Scatter Plot")
    if df.empty:
        st.write("No data to plot. Please upload a valid JSON file.")
    else:
        data = {'Signal1': df['signal1'][0], 'Signal2': df['signal2'][0], 'Signal3': df['signal3'][0]}
        chart_data = pd.DataFrame(data)

        st.line_chart(chart_data, color=["#ffaa00", "#ff0000", "#008000"])

    # Interactive Chart 2 - Bar Chart
    st.write("### Interactive Bar Chart")
    if df.empty:
        st.write("No data to plot. Please upload a valid JSON file.")
    else:
        data = {'Signal1': df['beforeCross1'][0], 'Signal2': df['beforeCross2'][0], 'Signal3': df['beforeCross3'][0]}
        chart_data = pd.DataFrame(data)

        st.line_chart(chart_data, color=["#ffaa00", "#ff0000", "#008000"])
