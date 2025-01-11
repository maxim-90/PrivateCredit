import streamlit as st
import pandas as pd
import pyxirr
import io
import os

# Set Streamlit to run in environments like Replit
os.environ['STREAMLIT_SERVER_PORT'] = '8080'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

# Title of the app
st.title("Private Markets Return Calculator")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the uploaded Excel file
    data = pd.read_excel(uploaded_file)

    # Show the uploaded data preview
    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    # Ensure required columns for cashflow, date, and type selection
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

    # Specify type column
    type_column = st.selectbox("Select the column for performance attribution grouping (e.g. cashflow type).", options=columns)
    # Define custom groups for attribution
    if type_column:
        st.write("Define your custom groups and map them to entries in the selected performance attribution column. Note, for a gross performance bridge with cashflow types, start with investment outlay and sales/principal repayments and market value/NAV. Afterwards build components by adding fees, dividends, interest, etc.")
        unique_types = data[type_column].unique()

        # Initialize session state for group mappings
        if "group_count" not in st.session_state:
            st.session_state.group_count = 1
            st.session_state.groups = [{"name": f"Group {i+1}", "types": []} for i in range(st.session_state.group_count)]

        # Add Group button
        if st.button("Add Group"):
            st.session_state.group_count += 1
            st.session_state.groups.append({"name": f"Group {st.session_state.group_count}", "types": []})

        # Render group inputs dynamically
        for i, group in enumerate(st.session_state.groups):
            group_name = st.text_input(f"Name of Group {i+1}", value=group["name"], key=f"group_name_{i}")
            selected_types = st.multiselect(
                f"Select types for {group_name}",
                options=unique_types,
                default=group["types"],
                key=f"group_types_{i}"
            )
            group["name"] = group_name
            group["types"] = selected_types

        # Map user-defined groups to a dictionary
        group_mapping = {group["name"]: group["types"] for group in st.session_state.groups}

        st.write("Your group mappings:")
        st.json(group_mapping)
        # Calculate Performance Attribution
        if st.button("Calculate IRR and MOIC Attribution"):
            if group_by_columns and cashflow_column and date_column:
                try:
                    # Initialize results
                    irr_attribution_results = []
                    moic_attribution_results = []

                    for group_keys, group_data in data.groupby(group_by_columns):
                        attribution_results = {}
                        previous_data = pd.DataFrame()

                        for group_name, types in group_mapping.items():
                            # Filter current group and concatenate with previous data
                            current_data = group_data[group_data["Type"].isin(types)]
                            combined_data = pd.concat([previous_data, current_data], ignore_index=True)

                            # Calculate IRR for combined data minus previous data
                            if not combined_data.empty and not previous_data.empty:
                                combined_cashflows = combined_data[cashflow_column]
                                combined_dates = combined_data[date_column]
                                previous_cashflows = previous_data[cashflow_column]
                                previous_dates = previous_data[date_column]
                                combined_irr = pyxirr.xirr(combined_dates, combined_cashflows)
                                previous_irr = pyxirr.xirr(previous_dates, previous_cashflows)
                                irr = combined_irr - previous_irr if combined_irr is not None and previous_irr is not None else combined_irr
                            elif not combined_data.empty:
                                combined_cashflows = combined_data[cashflow_column]
                                combined_dates = combined_data[date_column]
                                irr = pyxirr.xirr(combined_dates, combined_cashflows)
                            else:
                                irr = 0

                            # Calculate MOIC for combined data minus previous data
                            if not combined_data.empty and not previous_data.empty:
                                combined_positive = combined_data[cashflow_column][combined_data[cashflow_column] > 0].sum()
                                combined_negative = abs(combined_data[cashflow_column][combined_data[cashflow_column] < 0].sum())
                                combined_moic = combined_positive / combined_negative if combined_negative > 0 else None

                                previous_positive = previous_data[cashflow_column][previous_data[cashflow_column] > 0].sum()
                                previous_negative = abs(previous_data[cashflow_column][previous_data[cashflow_column] < 0].sum())
                                previous_moic = previous_positive / previous_negative if previous_negative > 0 else None

                                moic = combined_moic - previous_moic if previous_moic is not None else combined_moic
                            elif not combined_data.empty:
                                combined_positive = combined_data[cashflow_column][combined_data[cashflow_column] > 0].sum()
                                combined_negative = abs(combined_data[cashflow_column][combined_data[cashflow_column] < 0].sum())
                                moic = combined_positive / combined_negative if combined_negative > 0 else None
                            else:
                                moic = 0

                            attribution_results[group_name] = {"IRR": irr, "MOIC": moic}

                            # Update previous data
                            previous_data = combined_data

                        # Append results
                        irr_attribution_results.append(
                            {
                                **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                                **{f"{group_name} IRR": res["IRR"] for group_name, res in attribution_results.items()},
                            }
                        )
                        moic_attribution_results.append(
                            {
                                **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                                **{f"{group_name} MOIC": res["MOIC"] for group_name, res in attribution_results.items()},
                            }
                        )

                    # Convert to DataFrames
                    irr_attribution_df = pd.DataFrame(irr_attribution_results)
                    moic_attribution_df = pd.DataFrame(moic_attribution_results)

                    # Add totals column
                    irr_attribution_df["Total IRR"] = irr_attribution_df.iloc[:, len(group_by_columns):].sum(axis=1)
                    moic_attribution_df["Total MOIC"] = moic_attribution_df.iloc[:, len(group_by_columns):].sum(axis=1)

                    # Display results
                    st.write("Performance Attribution Results (IRR):")
                    st.dataframe(irr_attribution_df)

                    st.write("Performance Attribution Results (MOIC):")
                    st.dataframe(moic_attribution_df)

                    # Provide download option
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        irr_attribution_df.to_excel(writer, sheet_name="IRR Attribution Results", index=False)
                        moic_attribution_df.to_excel(writer, sheet_name="MOIC Attribution Results", index=False)
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
    else:
        st.warning("'Type' column not found in the uploaded data.")
