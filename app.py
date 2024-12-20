import streamlit as st
import pandas as pd
import pyxirr
import io

# Title of the app
st.title("Customizable IRR and MOIC Calculator")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the uploaded Excel file
    data = pd.read_excel(uploaded_file)

    # Show the uploaded data preview
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    # Ensure required columns for cashflow and date selection
    columns = data.columns.tolist()
    cashflow_column = st.selectbox("Select the column for cashflow", options=columns)
    date_column = st.selectbox("Select the column for dates", options=columns)

    # Show the "Group By" selection after cashflow and date column mapping
    group_by_columns = st.multiselect("Select columns to group by", options=columns, default=None)

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
                    irr_results[group_keys] = round(irr, 4)  # IRR in percentage with 4 decimals

                    # Calculate MOIC
                    positive_cashflows = cashflows[cashflows > 0].sum()
                    negative_cashflows = abs(cashflows[cashflows < 0].sum())
                    moic = round(positive_cashflows / negative_cashflows, 2) if negative_cashflows > 0 else None
                    moic_results[group_keys] = moic

                # Combine IRR and MOIC into a single DataFrame
                results_df = pd.DataFrame({
                    "IRR": pd.Series(irr_results),
                    "MOIC": pd.Series(moic_results),
                })
                results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=group_by_columns)

                # Display Results
                st.write("Results (IRR and MOIC):")
                st.dataframe(results_df)

                # Prepare the results for Excel export
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    results_df.to_excel(writer, sheet_name="Results")

                    # Access the xlsxwriter workbook and worksheet
                    workbook  = writer.book
                    worksheet = writer.sheets["Results"]

                    # Format the IRR column as percentage with 2 decimals in the UI
                    percent_format = workbook.add_format({'num_format': '0.00%'})
                    worksheet.set_column('B:B', 15, percent_format)

                output.seek(0)

                # Provide download button
                st.download_button(
                    label="Download Results as Excel",
                    data=output,
                    file_name="irr_moic_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Error calculating IRR or MOIC: {e}")

        else:
            st.warning("Please select all required columns for grouping, cashflow, and dates.")
