import streamlit as st
import pandas as pd
import pyxirr
import io
import os

# Set Streamlit to run in environments like Replit
os.environ['STREAMLIT_SERVER_PORT'] = '8080'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

# Title of the app
st.title("Private Credit Return Calculator")

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

    # Group by selection
    group_by_columns = st.multiselect("Select columns to group by", options=columns, default=None)

    # "Run IRR and MOIC" button
    if st.button("Run IRR and MOIC"):
        if group_by_columns and cashflow_column and date_column:
            st.write(f"Grouping by columns: {group_by_columns}")
            st.write(f"Cashflow column: {cashflow_column}")
            st.write(f"Date column: {date_column}")

            irr_results = {}
            moic_results = {}

            try:
                for group_keys, group_data in data.groupby(group_by_columns):
                    cashflows = group_data[cashflow_column]
                    dates = group_data[date_column]

                    # Calculate IRR
                    irr = pyxirr.xirr(dates, cashflows)
                    irr_results[group_keys] = round(irr, 4)

                    # Calculate MOIC
                    positive_cashflows = cashflows[cashflows > 0].sum()
                    negative_cashflows = abs(cashflows[cashflows < 0].sum())
                    moic = round(positive_cashflows / negative_cashflows, 2) if negative_cashflows > 0 else None
                    moic_results[group_keys] = moic

                results_df = pd.DataFrame({
                    "IRR": pd.Series(irr_results),
                    "MOIC": pd.Series(moic_results),
                })
                results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=group_by_columns)

                st.write("Results (IRR and MOIC):")
                st.dataframe(results_df)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    results_df.to_excel(writer, sheet_name="Results")

                output.seek(0)

                st.download_button(
                    label="Download Results as Excel",
                    data=output,
                    file_name="irr_moic_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Error calculating IRR or MOIC: {e}")

        else:
            st.warning("Please select columns to group by, and ensure cashflow and date columns are selected.")

    # "Calculate Performance Attribution" button
    if st.button("Calculate IRR Attribution"):
        if group_by_columns and cashflow_column and date_column:
            try:
                # Initialize results dictionary
                attribution_results = []

                # Group data by selected columns
                for group_keys, group_data in data.groupby(group_by_columns):
                    # Filters for different types
                    principal = group_data[group_data['Type'].isin(['Investment Outlay', 'Realised Principal', 'Outstanding Principal'])]
                    cash_interest = group_data[group_data['Type'].isin(['Cash Interest', 'Accrued Cash Interest'])]
                    upfront_fees = group_data[group_data['Type'].isin(['Upfront Fee'])]
                    pik_interest = group_data[group_data['Type'].isin(['Realised PIK', 'Accrued PIK'])]

                    # Concatenate DataFrames
                    principal_cash = pd.concat([principal, cash_interest], ignore_index=True)
                    principal_cash_fees = pd.concat([principal_cash, upfront_fees], ignore_index=True)
                    principal_cash_fees_pik = pd.concat([principal_cash_fees, pik_interest], ignore_index=True)

                    # Calculate contributions
                    cash_contribution = pyxirr.xirr(principal_cash[date_column], principal_cash[cashflow_column]) - \
                                        pyxirr.xirr(principal[date_column], principal[cashflow_column])
                    fee_contribution = pyxirr.xirr(principal_cash_fees[date_column], principal_cash_fees[cashflow_column]) - \
                                       pyxirr.xirr(principal_cash[date_column], principal_cash[cashflow_column])
                    pik_contribution = pyxirr.xirr(principal_cash_fees_pik[date_column], principal_cash_fees_pik[cashflow_column]) - \
                                       pyxirr.xirr(principal_cash_fees[date_column], principal_cash_fees[cashflow_column])

                    # Store results in a dictionary
                    attribution_results.append({
                        **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                        "Cash Contribution": cash_contribution,
                        "Fee Contribution": fee_contribution,
                        "PIK Contribution": pik_contribution
                    })

                # Convert results into a DataFrame
                attribution_df = pd.DataFrame(attribution_results)

                # Add a "Total" column
                attribution_df["Total"] = (
                    attribution_df["Cash Contribution"] +
                    attribution_df["Fee Contribution"] +
                    attribution_df["PIK Contribution"]
                )

                # Display results
                st.write("Performance Attribution Results:")
                st.dataframe(attribution_df)

                # Provide a download button for results
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    attribution_df.to_excel(writer, sheet_name="Attribution Results", index=False)
                output.seek(0)

                st.download_button(
                    label="Download Attribution Results as Excel",
                    data=output,
                    file_name="performance_attribution_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Error calculating performance attribution: {e}")
        else:
            st.warning("Please select columns to group by, and ensure cashflow and date columns are selected.")
