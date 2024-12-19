import streamlit as st
import pandas as pd
import pyxirr

# Title of the app
st.title("Customizable IRR and MOIC Calculator")

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

    # Add a Run button
    if st.button("Run IRR and MOIC"):
        if group_by_columns and cashflow_column and date_column:
            st.write(f"Grouping by columns: {group_by_columns}")
            st.write(f"Cashflow column: {cashflow_column}")
            st.write(f"Date column: {date_column}")

            # Group the data and calculate IRR and MOIC
            irr_results = {}
            moic_results = {}

            try:
                for group_keys, group_data in data.groupby(group_by_columns):
                    # Extract the relevant columns for calculations
                    cashflows = group_data[cashflow_column]
                    dates = group_data[date_column]

                    # Calculate IRR
                    irr = pyxirr.xirr(dates, cashflows)
                    irr_results[group_keys] = irr

                    # Calculate MOIC
                    positive_cashflows = cashflows[cashflows > 0].sum()
                    negative_cashflows = abs(cashflows[cashflows < 0].sum())
                    moic = positive_cashflows / negative_cashflows if negative_cashflows > 0 else None
                    moic_results[group_keys] = moic

                # Combine IRR and MOIC into a single DataFrame
                results_df = pd.DataFrame({
                    "IRR": pd.Series(irr_results),
                    "MOIC": pd.Series(moic_results),
                })
                results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=group_by_columns)

                # Display Results
                st.write("Results (IRR and MOIC):")
                st.write(results_df)

            except Exception as e:
                st.error(f"Error calculating IRR or MOIC: {e}")

        else:
            st.warning("Please select all required columns for grouping, cashflow, and dates.")
