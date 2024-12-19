import streamlit as st
import pandas as pd
import pyxirr

# Title of the app
st.title("Customizable IRR Calculator")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the uploaded Excel file
    data = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    # Select columns for grouping
    columns = data.columns.tolist()
    group_by_columns = st.multiselect("Select columns to group by", options=columns, default=None)

    # Ensure required columns for cashflow and date selection
    cashflow_column = st.selectbox("Select the column for cashflow", options=columns)
    date_column = st.selectbox("Select the column for dates", options=columns)

    if group_by_columns and cashflow_column and date_column:
        st.write(f"Grouping by columns: {group_by_columns}")
        st.write(f"Cashflow column: {cashflow_column}")
        st.write(f"Date column: {date_column}")

        # Group the data and calculate IRR
        irr_results = {}
        try:
            for group_keys, group_data in data.groupby(group_by_columns):
                # Extract the relevant columns for IRR calculation
                cashflows = group_data[cashflow_column]
                dates = group_data[date_column]
                irr = pyxirr.xirr(dates, cashflows)
                irr_results[group_keys] = irr

            # Convert IRR results to a DataFrame
            irr_df = pd.DataFrame.from_dict(
                irr_results, orient="index", columns=["IRR"]
            )
            irr_df.index = pd.MultiIndex.from_tuples(irr_df.index, names=group_by_columns)

            # Display IRR results
            st.write("IRR Results:")
            st.write(irr_df)

        except Exception as e:
            st.error(f"Error calculating IRR: {e}")

    else:
        st.warning("Please select all required columns for grouping, cashflow, and dates.")

