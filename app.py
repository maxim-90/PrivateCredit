import streamlit as st
import pandas as pd
import pyxirr
import io
import os
from datetime import datetime, date, timedelta
import numpy as np

# Set Streamlit to run in environments like Replit
os.environ['STREAMLIT_SERVER_PORT'] = '8080'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

# Title of the app
st.title("Private Markets Return Calculator")

# Left-hand navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Screens",
    [
        "IRR & MOIC",
        "Performance Attribution",
        "Loan Cashflow Forecast",
    ],
)

# File uploader (shown for analytics screens)
if section in ("IRR & MOIC", "Performance Attribution"):
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
else:
    uploaded_file = None

if uploaded_file and section in ("IRR & MOIC", "Performance Attribution"):
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
    
    # "Run IRR and MOIC" button (only on IRR screen)
    if section == "IRR & MOIC" and st.button("Run IRR and MOIC"):
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

    # Performance Attribution screen
    if section == "Performance Attribution":
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
                # Ensure defaults are valid options to avoid Streamlit exceptions
                _safe_default_types = [t for t in group["types"] if t in unique_types]
                selected_types = st.multiselect(
                    f"Select types for {group_name}",
                    options=unique_types,
                    default=_safe_default_types,
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
                                current_data = group_data[group_data[type_column].isin(types)]
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

                            # Append results for this outer group
                            irr_attribution_results.append(
                                {
                                    **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                                    **{f"{group_name} contribution": res["IRR"] for group_name, res in attribution_results.items()},
                                }
                            )
                            moic_attribution_results.append(
                                {
                                    **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                                    **{f"{group_name} contribution": res["MOIC"] for group_name, res in attribution_results.items()},
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

# --------------------------------------------
# Loan Cashflow Forecasting & Modelling
# --------------------------------------------
if section == "Loan Cashflow Forecast":
    st.header("Loan Cashflow Forecast")

    # Help text for users
    st.caption(
        "Build projected cashflows per loan with cash interest, PIK, interest frequency, base rate, accrued interest, and exit assumptions."
    )

    # Frequency mapping to months
    FREQUENCY_TO_MONTHS = {
        "Monthly": 1,
        "Quarterly": 3,
        "Semi-Annual": 6,
        "Annual": 12,
    }

    # Default loans template
    default_loans = pd.DataFrame(
        [
            {
                "Loan Name": "Loan A",
                "Currency": "USD",
                "Current Balance": 1_000_000.0,
                "Cash Interest % (over base)": 4.0,
                "PIK %": 2.0,
                "Base Rate Type": "SOFR",
                "Base Rate %": 5.0,
                "Interest Frequency": "Quarterly",
                "Last Coupon Date": pd.to_datetime(date.today() - timedelta(days=45)),
                "Exit Date": pd.to_datetime(date.today() + timedelta(days=365)),
                "Exit Price % of Par": 100.0,
            }
        ]
    )

    st.subheader("Loan Inputs")
    st.caption("Edit cells below. Add/remove rows as needed.")
    loans_df = st.data_editor(
        default_loans,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Cash Interest % (over base)": st.column_config.NumberColumn(format="%0.2f"),
            "PIK %": st.column_config.NumberColumn(format="%0.2f"),
            "Base Rate %": st.column_config.NumberColumn(format="%0.2f"),
            "Exit Price % of Par": st.column_config.NumberColumn(format="%0.2f"),
            "Current Balance": st.column_config.NumberColumn(format="% ,.2f"),
            "Last Coupon Date": st.column_config.DateColumn(),
            "Exit Date": st.column_config.DateColumn(),
            "Interest Frequency": st.column_config.SelectboxColumn(options=list(FREQUENCY_TO_MONTHS.keys())),
            "Base Rate Type": st.column_config.SelectboxColumn(options=["SOFR", "SONIA", "EURIBOR"]),
        },
    )

    as_of = st.date_input("As-of date for accruals", value=date.today())

    def month_end(dt: date) -> date:
        # Returns the last day of the month for a given date
        next_month = dt.replace(day=28) + timedelta(days=4)
        return (next_month - timedelta(days=next_month.day)).date()

    def add_months(dt: date, months: int) -> date:
        # Add months while preserving end-of-month logic
        year = dt.year + (dt.month - 1 + months) // 12
        month = (dt.month - 1 + months) % 12 + 1
        day = min(dt.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return date(year, month, day)

    def generate_schedule_for_loan(loan_row: pd.Series, as_of_date: date) -> pd.DataFrame:
        name = str(loan_row.get("Loan Name", "Loan"))
        currency = str(loan_row.get("Currency", "USD"))
        balance = float(loan_row.get("Current Balance", 0.0))
        cash_spread = float(loan_row.get("Cash Interest % (over base)", 0.0)) / 100.0
        pik_rate = float(loan_row.get("PIK %", 0.0)) / 100.0
        base_rate = float(loan_row.get("Base Rate %", 0.0)) / 100.0
        freq = str(loan_row.get("Interest Frequency", "Quarterly"))
        months = FREQUENCY_TO_MONTHS.get(freq, 3)
        last_coupon = loan_row.get("Last Coupon Date")
        exit_date = loan_row.get("Exit Date")
        exit_price_pct = float(loan_row.get("Exit Price % of Par", 100.0)) / 100.0

        if pd.isna(last_coupon):
            last_coupon = pd.to_datetime(as_of_date)
        if pd.isna(exit_date):
            exit_date = pd.to_datetime(as_of_date)

        last_coupon_date = pd.to_datetime(last_coupon).date()
        exit_dt = pd.to_datetime(exit_date).date()

        # Build period dates from the day after last coupon up to exit
        schedule_rows = []

        # Handle partial accrual from last coupon to as_of (if before next coupon)
        # Interest accrues daily on balance at (base + spread + PIK). Cash pays per frequency; PIK capitalizes at coupon dates.
        annual_rate_total = base_rate + cash_spread + pik_rate

        # Generate coupon dates
        coupon_date = add_months(last_coupon_date, months)
        while coupon_date < exit_dt:
            period_start = last_coupon_date
            period_end = coupon_date
            days = (period_end - period_start).days
            # Simple ACT/365 accrual
            period_interest_total = balance * annual_rate_total * (days / 365.0)
            period_cash_interest = balance * (base_rate + cash_spread) * (days / 365.0)
            period_pik_interest = period_interest_total - period_cash_interest
            balance += period_pik_interest  # Capitalize PIK

            schedule_rows.append({
                "Loan Name": name,
                "Currency": currency,
                "Date": period_end,
                "Type": "Coupon Payment",
                "Days": days,
                "Cash Interest": round(period_cash_interest, 2),
                "PIK Interest": round(period_pik_interest, 2),
                "Principal": 0.0,
                "Balance": round(balance, 2),
            })

            last_coupon_date = coupon_date
            coupon_date = add_months(coupon_date, months)

        # Final partial period from last coupon to exit
        if last_coupon_date < exit_dt:
            days = (exit_dt - last_coupon_date).days
            period_interest_total = balance * annual_rate_total * (days / 365.0)
            period_cash_interest = balance * (base_rate + cash_spread) * (days / 365.0)
            period_pik_interest = period_interest_total - period_cash_interest
            # Accrued interest: cash portion accrues but is not paid until exit; PIK accrues but is not capitalized until exit

            accrued_cash = period_cash_interest
            accrued_pik = period_pik_interest
            balance_at_exit_before_pik_cap = balance
            balance += accrued_pik  # Capitalize PIK at exit

            # Exit flows
            principal_repaid = balance_at_exit_before_pik_cap * exit_price_pct
            premium_or_discount = principal_repaid - balance_at_exit_before_pik_cap

            schedule_rows.append({
                "Loan Name": name,
                "Currency": currency,
                "Date": exit_dt,
                "Type": "Exit Accrual",
                "Days": days,
                "Cash Interest": round(accrued_cash, 2),
                "PIK Interest": round(accrued_pik, 2),
                "Principal": 0.0,
                "Balance": round(balance, 2),
            })

            schedule_rows.append({
                "Loan Name": name,
                "Currency": currency,
                "Date": exit_dt,
                "Type": "Exit Payment",
                "Days": 0,
                "Cash Interest": 0.0,
                "PIK Interest": 0.0,
                "Principal": round(principal_repaid, 2),
                "Balance": round(balance - principal_repaid, 2),
            })

        schedule_df = pd.DataFrame(schedule_rows)
        if not schedule_df.empty:
            schedule_df = schedule_df.sort_values(["Loan Name", "Date"]).reset_index(drop=True)
        return schedule_df


    if st.button("Generate Loan Cashflow Forecast"):
        try:
            schedules = []
            for _, row in loans_df.iterrows():
                schedules.append(generate_schedule_for_loan(row, as_of))
            full_schedule = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()

            if full_schedule.empty:
                st.warning("No schedule generated. Please check inputs.")
            else:
                st.subheader("Projected Cashflows")
                st.dataframe(full_schedule, use_container_width=True)

                # Summary per loan
                st.subheader("Summary by Loan")
                summary = (
                    full_schedule.groupby(["Loan Name", "Currency"], as_index=False)[["Cash Interest", "PIK Interest", "Principal"]]
                    .sum()
                )
                st.dataframe(summary, use_container_width=True)

                # Download
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                    full_schedule.to_excel(writer, sheet_name="Schedule", index=False)
                    summary.to_excel(writer, sheet_name="Summary", index=False)
                out.seek(0)
                st.download_button(
                    label="Download Forecast as Excel",
                    data=out,
                    file_name="loan_cashflow_forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as ex:
            st.error(f"Error generating forecast: {ex}")

