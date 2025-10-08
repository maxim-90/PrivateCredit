import streamlit as st
import pandas as pd
import pyxirr
import io
import os
from datetime import datetime, date, timedelta
import numpy as np
import requests
import json

# Set Streamlit to run in environments like Replit
os.environ['STREAMLIT_SERVER_PORT'] = '8080'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

# FRED API configuration
FRED_API_KEY = "2d2513d1b7c8d2e6d2d9cfde1d2ea6a6"
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

def fetch_fred_data(series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch data from FRED API for a given series ID."""
    try:
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'limit': 1000
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        response = requests.get(f"{FRED_BASE_URL}/series/observations", params=params)
        response.raise_for_status()
        
        data = response.json()
        if 'observations' not in data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data['observations'])
        if df.empty:
            return pd.DataFrame()
            
        # Convert date and value columns
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna(subset=['value'])
        
        return df[['date', 'value']].sort_values('date')
        
    except Exception as e:
        st.error(f"Error fetching FRED data for {series_id}: {str(e)}")
        return pd.DataFrame()

def get_treasury_curve(investment_date: date = None, exit_date: date = None) -> pd.DataFrame:
    """Fetch Treasury data and create 3M forward rates using discount factor bootstrapping."""
    try:
        # Fetch Treasury Constant Maturity Yields
        treasury_data = {}
        treasury_series = {
            '3M': 'DGS3MO',   # 3-Month Treasury
            '6M': 'DGS6MO',   # 6-Month Treasury  
            '1Y': 'DGS1',     # 1-Year Treasury
            '2Y': 'DGS2',     # 2-Year Treasury
            '5Y': 'DGS5',     # 5-Year Treasury
            '10Y': 'DGS10',   # 10-Year Treasury
            '30Y': 'DGS30'    # 30-Year Treasury
        }
        
        # Convert investment_date to string format for FRED API
        investment_date_str = investment_date.strftime('%Y-%m-%d') if investment_date else None
        
        # Calculate start date (90 days before investment date to ensure we have data)
        start_date_str = None
        if investment_date:
            start_date = investment_date - timedelta(days=90)
            start_date_str = start_date.strftime('%Y-%m-%d')
        
        for tenor, series_id in treasury_series.items():
            data = fetch_fred_data(series_id, start_date=start_date_str, end_date=investment_date_str)
            if not data.empty:
                # Get the rate as of the investment date (or closest available date before)
                investment_date_pd = pd.to_datetime(investment_date) if investment_date else pd.Timestamp.now()
                
                # Filter data to dates on or before investment date
                data_filtered = data[data['date'] <= investment_date_pd]
                
                if not data_filtered.empty:
                    # Use the rate closest to (but not after) the investment date
                    treasury_data[tenor] = data_filtered['value'].iloc[-1] / 100.0  # Convert to decimal
                else:
                    st.warning(f"No Treasury data available before investment date for {tenor}")
            else:
                st.warning(f"Could not fetch {tenor} Treasury data from FRED")
        
        if not treasury_data:
            st.error("Could not fetch any Treasury data from FRED")
            return pd.DataFrame()
            
        # Use the shortest available rate as our 3M proxy
        if '3M' in treasury_data:
            latest_3m = treasury_data['3M']
        elif '6M' in treasury_data:
            latest_3m = treasury_data['6M']
        else:
            latest_3m = list(treasury_data.values())[0]  # Use first available rate
        
        # Get rates for bootstrapping
        latest_6m = treasury_data.get('6M', latest_3m)
        latest_1y = treasury_data.get('1Y', latest_6m)
        latest_2y = treasury_data.get('2Y', latest_1y)
        
        # Create 3M forward rates using discount factor bootstrapping
        curve_start_date = investment_date if investment_date else date.today()
        curve_end_date = exit_date if exit_date else add_months(curve_start_date, 24)
        
        # Calculate the number of 3-month periods needed
        total_months = (curve_end_date.year - curve_start_date.year) * 12 + (curve_end_date.month - curve_start_date.month)
        num_quarters = (total_months + 2) // 3
        num_quarters = max(num_quarters, 1)
        
        # Proper bootstrapping methodology using linear interpolation
        import numpy as np
        
        # Define market maturities and yields (in years)
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        yields = np.array([
            latest_3m, latest_6m, latest_1y, latest_2y,
            treasury_data.get('5Y', latest_2y), 
            treasury_data.get('10Y', latest_2y), 
            treasury_data.get('30Y', latest_2y)
        ])
        
        # Create discount factors from zero rates
        discount_factors = 1 / (1 + yields) ** maturities
        
        # Function to interpolate zero rates linearly
        def zero_rate_interp(t, maturities, yields):
            if t in maturities:
                return yields[maturities == t][0]
            else:
                for i in range(len(maturities)-1):
                    if maturities[i] < t < maturities[i+1]:
                        t1, t2 = maturities[i], maturities[i+1]
                        r1, r2 = yields[i], yields[i+1]
                        r_t = r1 + (r2 - r1) * (t - t1) / (t2 - t1)
                        return r_t
                # If t beyond last maturity, extrapolate using last interval
                t1, t2 = maturities[-2], maturities[-1]
                r1, r2 = yields[-2], yields[-1]
                r_t = r2 + (r2 - r1) / (t2 - t1) * (t - t2)
                return r_t
        
        # Generate 3-month forward rates
        tenor = 0.25  # 3 months
        max_maturity = (num_quarters * 3) / 12.0  # Convert quarters to years
        forward_times = np.arange(tenor, max_maturity + tenor, tenor)
        
        forward_periods = []
        forward_rates = []
        
        for i, t in enumerate(forward_times):
            t_prev = t - tenor
            
            # Calculate dates
            forward_start_date = add_months(curve_start_date, i * 3)
            forward_end_date = add_months(curve_start_date, (i + 1) * 3)
            
            if forward_start_date >= curve_end_date:
                break
                
            if forward_end_date > curve_end_date:
                forward_end_date = curve_end_date
            
            forward_periods.append({
                'start_date': forward_start_date,
                'end_date': forward_end_date,
                'period_months': 3
            })
            
            # Interpolate zero rates and compute discount factors
            if t_prev == 0:
                df_prev = 1.0
            else:
                r_prev = zero_rate_interp(t_prev, maturities, yields)
                df_prev = 1 / (1 + r_prev) ** t_prev
            
            r_t = zero_rate_interp(t, maturities, yields)
            df_t = 1 / (1 + r_t) ** t
            
            # Forward rate formula: f = (df_prev / df_t)^(1/tenor) - 1
            f = (df_prev / df_t) ** (1/tenor) - 1
            
            forward_rates.append(round(f, 6))
            
        return pd.DataFrame({
            'forward_start': [p['start_date'] for p in forward_periods],
            'forward_end': [p['end_date'] for p in forward_periods],
            'rate': forward_rates
        })
        
    except Exception as e:
        st.error(f"Error creating Treasury curve: {str(e)}")
        return pd.DataFrame()

def get_forward_rate_for_period(period_start_date: date, treasury_curve: pd.DataFrame = None) -> float:
    """Get the 3M forward rate for a period starting at the given date."""
    if treasury_curve is None or treasury_curve.empty:
        return 0.05  # 5% default
    
    target_datetime = pd.to_datetime(period_start_date)
    
    # Find the appropriate 3M forward rate for this period start date
    for i, row in treasury_curve.iterrows():
        forward_start = pd.to_datetime(row['forward_start'])
        forward_end = pd.to_datetime(row['forward_end'])
        
        if forward_start <= target_datetime < forward_end or target_datetime == forward_start:
            return row['rate']
    
    # If target date is before the first forward period, use the first rate
    if target_datetime < pd.to_datetime(treasury_curve['forward_start'].iloc[0]):
        return treasury_curve['rate'].iloc[0]
    
    # If target date is after the last forward period, use the last rate
    if target_datetime >= pd.to_datetime(treasury_curve['forward_end'].iloc[-1]):
        return treasury_curve['rate'].iloc[-1]
    
    # If we're between forward periods, interpolate
    for i in range(len(treasury_curve) - 1):
        current_end = pd.to_datetime(treasury_curve['forward_end'].iloc[i])
        next_start = pd.to_datetime(treasury_curve['forward_start'].iloc[i + 1])
        
        if current_end <= target_datetime <= next_start:
            rate1 = treasury_curve['rate'].iloc[i]
            rate2 = treasury_curve['rate'].iloc[i + 1]
            total_days = (next_start - current_end).days
            target_days = (target_datetime - current_end).days
            
            if total_days == 0:
                return rate1
                
            factor = target_days / total_days
            return rate1 + factor * (rate2 - rate1)
    
    return treasury_curve['rate'].iloc[-1]

def add_months(dt: date, months: int) -> date:
    # Add months while preserving end-of-month logic
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = min(
        dt.day,
        [
            31,
            29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ][month - 1],
    )
    return date(year, month, day)

# Title of the app
st.title("Private Markets Return Calculator")

# Left-hand navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Screens",
    [
        "IRR & MOIC",
        "Performance Attribution",
        "Cashflow Forecasting",
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

            # Attribution method selection
            attribution_method = st.selectbox(
                "Attribution method",
                options=["Marginal", "Allocation-based"],
                index=0,
            )

            def safe_xirr(xirr_dates: pd.Series, xirr_cashflows: pd.Series):
                try:
                    return pyxirr.xirr(pd.to_datetime(xirr_dates), xirr_cashflows)
                except Exception:
                    return None

            # Capital bucket selection (used by Allocation-based variant)
            capital_bucket = None
            if attribution_method == "Allocation-based":
                group_names = list(group_mapping.keys())
                # Try to find a sensible default bucket name
                default_cap_idx = 0
                for i, g in enumerate(group_names):
                    g_lower = str(g).lower()
                    if "capital" in g_lower or "principal" in g_lower:
                        default_cap_idx = i
                        break
                capital_bucket = st.selectbox(
                    "Select capital bucket (principal outlay/repayments)",
                    options=group_names,
                    index=default_cap_idx,
                )
            # Calculate Performance Attribution
            if st.button("Calculate IRR and MOIC Attribution"):
                if group_by_columns and cashflow_column and date_column:
                    try:
                        if attribution_method == "Marginal":
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

                                    # IRR contribution (marginal)
                                    if not combined_data.empty and not previous_data.empty:
                                        combined_cashflows = combined_data[cashflow_column]
                                        combined_dates = combined_data[date_column]
                                        previous_cashflows = previous_data[cashflow_column]
                                        previous_dates = previous_data[date_column]
                                        combined_irr = safe_xirr(combined_dates, combined_cashflows)
                                        previous_irr = safe_xirr(previous_dates, previous_cashflows)
                                        irr_contribution = combined_irr - previous_irr if (combined_irr is not None and previous_irr is not None) else combined_irr
                                    elif not combined_data.empty:
                                        combined_cashflows = combined_data[cashflow_column]
                                        combined_dates = combined_data[date_column]
                                        irr_contribution = safe_xirr(combined_dates, combined_cashflows)
                                    else:
                                        irr_contribution = 0

                                    # MOIC contribution (marginal)
                                    if not combined_data.empty and not previous_data.empty:
                                        combined_positive = combined_data[cashflow_column][combined_data[cashflow_column] > 0].sum()
                                        combined_negative = abs(combined_data[cashflow_column][combined_data[cashflow_column] < 0].sum())
                                        combined_moic = combined_positive / combined_negative if combined_negative > 0 else None

                                        previous_positive = previous_data[cashflow_column][previous_data[cashflow_column] > 0].sum()
                                        previous_negative = abs(previous_data[cashflow_column][previous_data[cashflow_column] < 0].sum())
                                        previous_moic = previous_positive / previous_negative if previous_negative > 0 else None

                                        moic_contribution = (combined_moic - previous_moic) if (combined_moic is not None and previous_moic is not None) else combined_moic
                                    elif not combined_data.empty:
                                        combined_positive = combined_data[cashflow_column][combined_data[cashflow_column] > 0].sum()
                                        combined_negative = abs(combined_data[cashflow_column][combined_data[cashflow_column] < 0].sum())
                                        moic_contribution = combined_positive / combined_negative if combined_negative > 0 else None
                                    else:
                                        moic_contribution = 0

                                    attribution_results[group_name] = {"IRR": irr_contribution, "MOIC": moic_contribution}
                                    previous_data = combined_data.copy()

                                irr_attribution_results.append({
                                    **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                                    **{f"{group_name} contribution": res["IRR"] for group_name, res in attribution_results.items()},
                                })
                                moic_attribution_results.append({
                                    **{col: group_keys[i] for i, col in enumerate(group_by_columns)},
                                    **{f"{group_name} contribution": res["MOIC"] for group_name, res in attribution_results.items()},
                                })

                            irr_attribution_df = pd.DataFrame(irr_attribution_results)
                            moic_attribution_df = pd.DataFrame(moic_attribution_results)
                            irr_attribution_df["Total IRR"] = irr_attribution_df.iloc[:, len(group_by_columns):].sum(axis=1)
                            moic_attribution_df["Total MOIC"] = moic_attribution_df.iloc[:, len(group_by_columns):].sum(axis=1)
                        else:
                            # Two-step allocation-based attribution
                            # Step 1: Capital IRR from selected capital bucket
                            # Step 2: Allocate remaining IRR (portfolio - capital) across other buckets by discounted cashflow magnitude
                            irr_rows = []
                            moic_rows = []

                            for group_keys, group_data in data.groupby(group_by_columns):
                                portfolio_irr = safe_xirr(group_data[date_column], group_data[cashflow_column])
                                attribution_results = {}
                                if portfolio_irr is not None and capital_bucket is not None:
                                    # Capital IRR from capital bucket only
                                    cap_types = group_mapping.get(capital_bucket, [])
                                    cap_cf = group_data[group_data[type_column].isin(cap_types)]
                                    capital_irr = safe_xirr(cap_cf[date_column], cap_cf[cashflow_column]) if not cap_cf.empty else 0

                                    # Remaining IRR to allocate
                                    capital_irr = capital_irr if capital_irr is not None else 0
                                    delta_irr = portfolio_irr - capital_irr

                                    # Discount remaining groups at delta_irr and allocate proportionally
                                    t0 = pd.to_datetime(group_data[date_column]).min()
                                    # Prevent zero/negative discount rate blowing up: use a tiny epsilon
                                    alloc_rate = delta_irr if abs(delta_irr) > 1e-9 else 1e-9
                                    discounted_by_group = {}
                                    for group_name, types in group_mapping.items():
                                        if group_name == capital_bucket:
                                            continue
                                        group_cf = group_data[group_data[type_column].isin(types)]
                                        if not group_cf.empty:
                                            t_years = (pd.to_datetime(group_cf[date_column]) - t0).dt.days / 365.0
                                            discounted_sum = (group_cf[cashflow_column] / ((1 + alloc_rate) ** t_years)).sum()
                                            discounted_by_group[group_name] = float(discounted_sum)
                                        else:
                                            discounted_by_group[group_name] = 0.0

                                    total_abs = sum(abs(v) for v in discounted_by_group.values())

                                    # Set capital contribution first
                                    attribution_results[capital_bucket] = {"IRR": capital_irr, "MOIC": None}

                                    if total_abs == 0:
                                        # No meaningful remaining buckets; assign zero to others
                                        for group_name in group_mapping.keys():
                                            if group_name == capital_bucket:
                                                continue
                                            attribution_results[group_name] = {"IRR": 0, "MOIC": None}
                                    else:
                                        weights = {g: (abs(v) / total_abs) for g, v in discounted_by_group.items()}
                                        contributions = {g: delta_irr * w for g, w in weights.items()}
                                        residual = delta_irr - sum(contributions.values())
                                        if contributions:
                                            max_g = max(weights, key=weights.get)
                                            contributions[max_g] += residual
                                        for g, c in contributions.items():
                                            attribution_results[g] = {"IRR": c, "MOIC": None}
                                else:
                                    for group_name in group_mapping.keys():
                                        attribution_results[group_name] = {"IRR": 0, "MOIC": None}

                                # Build rows: IRR from allocation-based scheme; MOIC should mirror Marginal method output
                                irr_row = {col: group_keys[i] for i, col in enumerate(group_by_columns)}
                                irr_row.update({f"{name} contribution": vals["IRR"] for name, vals in attribution_results.items()})

                                # Compute MOIC contributions using Marginal definition to keep parity across methods
                                moic_attrib = {}
                                prev_df = pd.DataFrame()
                                for g_name, types in group_mapping.items():
                                    cur_df = group_data[group_data[type_column].isin(types)]
                                    comb_df = pd.concat([prev_df, cur_df], ignore_index=True)
                                    if not comb_df.empty and not prev_df.empty:
                                        comb_pos = comb_df[cashflow_column][comb_df[cashflow_column] > 0].sum()
                                        comb_neg = abs(comb_df[cashflow_column][comb_df[cashflow_column] < 0].sum())
                                        comb_moic = (comb_pos / comb_neg) if comb_neg > 0 else None
                                        prev_pos = prev_df[cashflow_column][prev_df[cashflow_column] > 0].sum()
                                        prev_neg = abs(prev_df[cashflow_column][prev_df[cashflow_column] < 0].sum())
                                        prev_moic = (prev_pos / prev_neg) if prev_neg > 0 else None
                                        moic_contrib = (comb_moic - prev_moic) if (comb_moic is not None and prev_moic is not None) else comb_moic
                                    elif not comb_df.empty:
                                        comb_pos = comb_df[cashflow_column][comb_df[cashflow_column] > 0].sum()
                                        comb_neg = abs(comb_df[cashflow_column][comb_df[cashflow_column] < 0].sum())
                                        moic_contrib = (comb_pos / comb_neg) if comb_neg > 0 else None
                                    else:
                                        moic_contrib = 0
                                    moic_attrib[g_name] = moic_contrib
                                    prev_df = comb_df

                                moic_row = {col: group_keys[i] for i, col in enumerate(group_by_columns)}
                                moic_row.update({f"{name} contribution": moic_attrib[name] for name in group_mapping.keys()})

                                irr_rows.append(irr_row)
                                moic_rows.append(moic_row)

                            irr_attribution_df = pd.DataFrame(irr_rows)
                            moic_attribution_df = pd.DataFrame(moic_rows)

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
# Cashflow Forecasting - Asset Class Picker
# --------------------------------------------
if section == "Cashflow Forecasting":
    st.header("Cashflow Forecasting")
    asset_class = st.selectbox("Asset class", ["Private Credit", "Real Estate", "Private Equity", "Infrastructure"]) 

    # --------------------------------------------
    # Real Estate Cashflow Forecasting
    # --------------------------------------------
    if asset_class == "Real Estate":
        st.caption("Model property-level cashflows with rental income, operating expenses, and exit proceeds.")

        # Property type options
        PROPERTY_TYPES = ["Office", "Retail", "Industrial", "Multifamily", "Hotel", "Mixed-Use", "Other"]
        PAYMENT_FREQ = ["Monthly", "Quarterly", "Semi-Annual", "Annual"]
        FREQ_TO_MONTHS_RE = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6, "Annual": 12}

        default_re = pd.DataFrame([
            {
                "Property Name": "Property A",
                "Property Type": "Office",
                "Currency": "USD",
                "Acquisition Value": 10_000_000.0,
                "Acquisition Date": pd.to_datetime(date.today() - timedelta(days=90)),
                "Exit Date": pd.to_datetime(date.today() + timedelta(days=1825)),  # 5 years
                "Exit Cap Rate %": 6.5,
                "Base NOI (Annual)": 650_000.0,
                "NOI Growth Rate %": 2.5,
                "Occupancy Rate %": 95.0,
                "Other Income % of Revenue": 3.0,
                "OpEx % of Revenue": 35.0,
                "OpEx Growth Rate %": 2.0,
                "CapEx Reserve % of Revenue": 5.0,
                "Property Mgmt Fee % of NOI": 3.0,
                "Rental Payment Frequency": "Quarterly",
                "Leverage %": 60.0,
                "Interest Rate %": 5.0,
                "Interest Only Period (Years)": 2,
                "Amortization Period (Years)": 25,
            }
        ])

        st.subheader("Real Estate Inputs")
        st.caption("Edit cells below. Add/remove rows as needed.")
        re_df = st.data_editor(
            default_re,
            num_rows="dynamic",
            width='stretch',
            column_config={
                "Property Type": st.column_config.SelectboxColumn(options=PROPERTY_TYPES),
                "Acquisition Value": st.column_config.NumberColumn(),
                "Base NOI (Annual)": st.column_config.NumberColumn(),
                "Exit Cap Rate %": st.column_config.NumberColumn(format="%.2f"),
                "NOI Growth Rate %": st.column_config.NumberColumn(format="%.2f"),
                "Occupancy Rate %": st.column_config.NumberColumn(format="%.1f"),
                "Other Income % of Revenue": st.column_config.NumberColumn(format="%.2f"),
                "OpEx % of Revenue": st.column_config.NumberColumn(format="%.2f"),
                "OpEx Growth Rate %": st.column_config.NumberColumn(format="%.2f"),
                "CapEx Reserve % of Revenue": st.column_config.NumberColumn(format="%.2f"),
                "Property Mgmt Fee % of NOI": st.column_config.NumberColumn(format="%.2f"),
                "Leverage %": st.column_config.NumberColumn(format="%.1f"),
                "Interest Rate %": st.column_config.NumberColumn(format="%.2f"),
                "Interest Only Period (Years)": st.column_config.NumberColumn(format="%.0f"),
                "Amortization Period (Years)": st.column_config.NumberColumn(format="%.0f"),
                "Acquisition Date": st.column_config.DateColumn(),
                "Exit Date": st.column_config.DateColumn(),
                "Rental Payment Frequency": st.column_config.SelectboxColumn(options=PAYMENT_FREQ),
            },
        )

        def generate_re_schedule(row: pd.Series) -> pd.DataFrame:
            name = str(row.get("Property Name", "Property"))
            currency = str(row.get("Currency", "USD"))
            acq_value = float(row.get("Acquisition Value", 0))
            acq_date = pd.to_datetime(row.get("Acquisition Date")).date()
            exit_date = pd.to_datetime(row.get("Exit Date")).date()
            exit_cap = float(row.get("Exit Cap Rate %", 6.5)) / 100.0
            base_noi = float(row.get("Base NOI (Annual)", 0))
            noi_growth = float(row.get("NOI Growth Rate %", 0)) / 100.0
            occupancy = float(row.get("Occupancy Rate %", 100)) / 100.0
            other_income_pct = float(row.get("Other Income % of Revenue", 0)) / 100.0
            opex_pct = float(row.get("OpEx % of Revenue", 0)) / 100.0
            opex_growth = float(row.get("OpEx Growth Rate %", 0)) / 100.0
            capex_pct = float(row.get("CapEx Reserve % of Revenue", 0)) / 100.0
            mgmt_fee_pct = float(row.get("Property Mgmt Fee % of NOI", 0)) / 100.0
            freq = str(row.get("Rental Payment Frequency", "Quarterly"))
            months = FREQ_TO_MONTHS_RE.get(freq, 3)
            
            leverage_pct = float(row.get("Leverage %", 0)) / 100.0
            interest_rate = float(row.get("Interest Rate %", 0)) / 100.0
            io_period_years = float(row.get("Interest Only Period (Years)", 0))
            amort_years = float(row.get("Amortization Period (Years)", 25))

            # Debt calculation
            initial_debt = acq_value * leverage_pct
            debt_balance = initial_debt

            schedule = []
            
            # Initial acquisition cashflow
            equity_investment = acq_value * (1 - leverage_pct)
            schedule.append({
                "Property Name": name,
                "Currency": currency,
                "Date": acq_date,
                "Type": "Acquisition",
                "Days": 0,
                "Gross Revenue": 0,
                "Operating Expenses": 0,
                "CapEx Reserve": 0,
                "Management Fee": 0,
                "NOI After Fees": 0,
                "Interest Payment": 0,
                "Principal Payment": 0,
                "Cash to Equity": round(-equity_investment, 2),
                "Debt Balance": round(initial_debt, 2),
                "Net Cashflow": round(-equity_investment, 2),
            })
            
            current_date = acq_date
            year_counter = 0

            # Generate periodic cashflows
            while current_date < exit_date:
                next_date = add_months(current_date, months)
                if next_date > exit_date:
                    next_date = exit_date

                # Calculate period in years
                days_in_period = (next_date - current_date).days
                years_elapsed = (current_date - acq_date).days / 365.25
                
                # Revenue: NOI grows annually, adjusted for occupancy
                annual_noi = base_noi * ((1 + noi_growth) ** years_elapsed) * occupancy
                period_noi = annual_noi * (days_in_period / 365.25)
                
                # Other income
                other_income = period_noi * other_income_pct
                gross_revenue = period_noi + other_income
                
                # Operating expenses
                opex = gross_revenue * opex_pct * ((1 + opex_growth) ** years_elapsed)
                capex = gross_revenue * capex_pct
                net_noi = gross_revenue - opex - capex
                mgmt_fee = net_noi * mgmt_fee_pct
                
                noi_after_fees = net_noi - mgmt_fee
                
                # Debt service
                if debt_balance > 0:
                    period_interest = debt_balance * interest_rate * (days_in_period / 365.25)
                    
                    # Check if we're past interest-only period
                    if years_elapsed > io_period_years:
                        # Amortizing payment
                        remaining_amort_periods = (amort_years - years_elapsed) * (12 / months)
                        if remaining_amort_periods > 0:
                            # Calculate periodic payment using amortization formula
                            periodic_rate = interest_rate * (months / 12)
                            periodic_payment = debt_balance * (periodic_rate * (1 + periodic_rate)**remaining_amort_periods) / ((1 + periodic_rate)**remaining_amort_periods - 1)
                            principal_payment = periodic_payment - period_interest
                            debt_balance -= principal_payment
                        else:
                            principal_payment = debt_balance
                            debt_balance = 0
                    else:
                        # Interest only
                        principal_payment = 0
                        
                    total_debt_service = period_interest + principal_payment
                else:
                    period_interest = 0
                    principal_payment = 0
                    total_debt_service = 0
                
                # Cash to equity
                cash_to_equity = noi_after_fees - total_debt_service
                
                schedule.append({
                    "Property Name": name,
                    "Currency": currency,
                    "Date": next_date,
                    "Type": "Operating Period",
                    "Days": days_in_period,
                    "Gross Revenue": round(gross_revenue, 2),
                    "Operating Expenses": round(opex, 2),
                    "CapEx Reserve": round(capex, 2),
                    "Management Fee": round(mgmt_fee, 2),
                    "NOI After Fees": round(noi_after_fees, 2),
                    "Interest Payment": round(period_interest, 2),
                    "Principal Payment": round(principal_payment, 2),
                    "Cash to Equity": round(cash_to_equity, 2),
                    "Debt Balance": round(debt_balance, 2),
                    "Net Cashflow": round(cash_to_equity, 2),
                })
                
                current_date = next_date
                year_counter += 1

            # Exit cashflow
            years_at_exit = (exit_date - acq_date).days / 365.25
            final_noi = base_noi * ((1 + noi_growth) ** years_at_exit) * occupancy
            exit_value = final_noi / exit_cap
            
            # Calculate accrued interest from last operating period to exit
            # Find the last operating period date
            last_operating_year = current_date.year if current_date.year < exit_date.year else exit_date.year - 1
            last_operating_date = date(last_operating_year, 12, 31)
            if last_operating_date >= exit_date:
                # If exit is in the same year as last operating period, use last operating date
                last_operating_date = current_date
            
            # Calculate interest accrued from last period to exit
            days_since_last_period = (exit_date - last_operating_date).days
            accrued_interest = debt_balance * interest_rate * (days_since_last_period / 365.25)
            
            # Total debt payment includes principal plus accrued interest
            total_debt_payment = debt_balance + accrued_interest
            exit_proceeds = exit_value - total_debt_payment
            
            schedule.append({
                "Property Name": name,
                "Currency": currency,
                "Date": exit_date,
                "Type": "Exit Sale",
                "Days": 0,
                "Gross Revenue": 0,
                "Operating Expenses": 0,
                "CapEx Reserve": 0,
                "Management Fee": 0,
                "NOI After Fees": round(final_noi, 2),
                "Interest Payment": round(accrued_interest, 2),
                "Principal Payment": round(total_debt_payment, 2),
                "Cash to Equity": round(exit_proceeds, 2),
                "Debt Balance": 0,
                "Net Cashflow": round(exit_proceeds, 2),
            })

            return pd.DataFrame(schedule)

        if st.button("Generate Real Estate Forecast"):
            try:
                schedules = []
                for _, row in re_df.iterrows():
                    schedules.append(generate_re_schedule(row))
                full_schedule = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()

                if full_schedule.empty:
                    st.warning("No schedule generated. Please check inputs.")
                else:
                    st.subheader("Projected Cashflows")
                    st.dataframe(full_schedule, width='stretch')

                    # Summary
                    st.subheader("Summary by Property")
                    summary = full_schedule.groupby(["Property Name", "Currency"], as_index=False)[
                        ["Gross Revenue", "Operating Expenses", "CapEx Reserve", "Management Fee", 
                         "Interest Payment", "Principal Payment", "Cash to Equity", "Net Cashflow"]
                    ].sum()
                    
                    # Add returns calculation
                    for idx, row_sum in summary.iterrows():
                        property_name = row_sum["Property Name"]
                        prop_schedule = full_schedule[full_schedule["Property Name"] == property_name]
                        dates = pd.to_datetime(prop_schedule["Date"])
                        cashflows = prop_schedule["Net Cashflow"]
                        try:
                            irr = pyxirr.xirr(dates, cashflows)
                            summary.at[idx, "IRR %"] = round(irr * 100, 2)
                            
                            invested = abs(cashflows[cashflows < 0].sum())
                            returned = cashflows[cashflows > 0].sum()
                            moic = returned / invested if invested > 0 else 0
                            summary.at[idx, "MOIC"] = round(moic, 2)
                        except:
                            summary.at[idx, "IRR %"] = None
                            summary.at[idx, "MOIC"] = None
                    
                    # Format MOIC column to display with 2 decimal places
                    if "MOIC" in summary.columns:
                        summary["MOIC"] = summary["MOIC"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                    st.dataframe(summary, width='stretch')

                    # Download
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                        full_schedule.to_excel(writer, sheet_name="Schedule", index=False)
                        summary.to_excel(writer, sheet_name="Summary", index=False)
                    out.seek(0)
                    st.download_button(
                        label="Download Real Estate Forecast as Excel",
                        data=out,
                        file_name="real_estate_forecast.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as ex:
                st.error(f"Error generating real estate forecast: {ex}")

    # --------------------------------------------
    # Private Equity Cashflow Forecasting
    # --------------------------------------------
    if asset_class == "Private Equity":
        st.caption("Model portfolio company value creation with EBITDA growth, multiple expansion, and debt paydown.")

        SECTORS = ["Technology", "Healthcare", "Industrials", "Consumer", "Financial Services", "Energy", "Other"]
        
        default_pe = pd.DataFrame([
            {
                "Company Name": "Portfolio Co A",
                "Sector": "Technology",
                "Currency": "USD",
                "Equity Investment": 50_000_000.0,
                "Entry Date": pd.to_datetime(date.today() - timedelta(days=180)),
                "Exit Date": pd.to_datetime(date.today() + timedelta(days=1460)),  # 4 years
                "Ownership %": 100.0,
                "Entry EBITDA": 20_000_000.0,
                "Entry EV/EBITDA Multiple": 10.0,
                "Exit EV/EBITDA Multiple": 12.0,
                "EBITDA Growth Rate %": 8.0,
                "Entry Debt": 50_000_000.0,
                "Annual Debt Paydown": 5_000_000.0,
                "Interest Rate %": 6.0,
                "FCF Conversion %": 60.0,
                "Dividend Payout Ratio %": 0.0,
                "Management Fee % of NAV": 2.0,
                "Transaction Costs %": 2.0,
            }
        ])

        st.subheader("Private Equity Inputs")
        st.caption("Edit cells below. Add/remove rows as needed.")
        pe_df = st.data_editor(
            default_pe,
            num_rows="dynamic",
            width='stretch',
            column_config={
                "Sector": st.column_config.SelectboxColumn(options=SECTORS),
                "Equity Investment": st.column_config.NumberColumn(),
                "Entry EBITDA": st.column_config.NumberColumn(),
                "Entry EV/EBITDA Multiple": st.column_config.NumberColumn(format="%.1fx"),
                "Exit EV/EBITDA Multiple": st.column_config.NumberColumn(format="%.1fx"),
                "EBITDA Growth Rate %": st.column_config.NumberColumn(format="%.2f"),
                "Entry Debt": st.column_config.NumberColumn(),
                "Annual Debt Paydown": st.column_config.NumberColumn(),
                "Interest Rate %": st.column_config.NumberColumn(format="%.2f"),
                "FCF Conversion %": st.column_config.NumberColumn(format="%.1f"),
                "Ownership %": st.column_config.NumberColumn(format="%.1f"),
                "Dividend Payout Ratio %": st.column_config.NumberColumn(format="%.1f"),
                "Management Fee % of NAV": st.column_config.NumberColumn(format="%.2f"),
                "Transaction Costs %": st.column_config.NumberColumn(format="%.2f"),
                "Entry Date": st.column_config.DateColumn(),
                "Exit Date": st.column_config.DateColumn(),
            },
        )

        def generate_pe_schedule(row: pd.Series) -> pd.DataFrame:
            name = str(row.get("Company Name", "Company"))
            currency = str(row.get("Currency", "USD"))
            equity_inv = float(row.get("Equity Investment", 0))
            entry_date = pd.to_datetime(row.get("Entry Date")).date()
            exit_date = pd.to_datetime(row.get("Exit Date")).date()
            ownership = float(row.get("Ownership %", 100)) / 100.0
            entry_ebitda = float(row.get("Entry EBITDA", 0))
            entry_multiple = float(row.get("Entry EV/EBITDA Multiple", 10))
            exit_multiple = float(row.get("Exit EV/EBITDA Multiple", 12))
            ebitda_growth = float(row.get("EBITDA Growth Rate %", 0)) / 100.0
            entry_debt = float(row.get("Entry Debt", 0))
            annual_paydown = float(row.get("Annual Debt Paydown", 0))
            interest_rate = float(row.get("Interest Rate %", 0)) / 100.0
            fcf_conversion = float(row.get("FCF Conversion %", 60)) / 100.0
            dividend_ratio = float(row.get("Dividend Payout Ratio %", 0)) / 100.0
            mgmt_fee_pct = float(row.get("Management Fee % of NAV", 0)) / 100.0
            txn_costs_pct = float(row.get("Transaction Costs %", 0)) / 100.0

            schedule = []
            debt_balance = entry_debt
            entry_ev = entry_ebitda * entry_multiple
            entry_equity_value = entry_ev - entry_debt
            
            # Entry transaction
            txn_costs = equity_inv * txn_costs_pct
            schedule.append({
                "Company Name": name,
                "Currency": currency,
                "Date": entry_date,
                "Type": "Entry Investment",
                "EBITDA": round(entry_ebitda, 2),
                "EV": round(entry_ev, 2),
                "Debt": round(debt_balance, 2),
                "Equity Value": round(entry_equity_value, 2),
                "Investor Equity Cashflow": round(-equity_inv, 2),
                "Transaction Costs": round(-txn_costs, 2),
                "Management Fees": 0,
                "Interest Expense": 0,
                "Debt Paydown": 0,
                "Net Cashflow": round(-equity_inv - txn_costs, 2),
            })

            # Annual periods
            current_year = entry_date.year
            exit_year = exit_date.year
            years_held = (exit_date - entry_date).days / 365.25

            for year in range(current_year + 1, exit_year + 1):
                year_end = date(year, 12, 31)
                if year_end > exit_date:
                    year_end = exit_date
                    
                years_from_entry = (year_end - entry_date).days / 365.25
                
                # EBITDA projection
                current_ebitda = entry_ebitda * ((1 + ebitda_growth) ** years_from_entry)
                
                # Current valuation (mark-to-market)
                # Assume linear multiple expansion
                current_multiple = entry_multiple + (exit_multiple - entry_multiple) * (years_from_entry / years_held)
                current_ev = current_ebitda * current_multiple
                
                # Interest expense
                annual_interest = debt_balance * interest_rate
                
                # Debt paydown
                actual_paydown = min(annual_paydown, debt_balance)
                debt_balance -= actual_paydown
                
                # Current equity value
                current_equity_value = current_ev - debt_balance
                
                # Management fees
                mgmt_fee = current_equity_value * mgmt_fee_pct
                
                # Dividends (if any)
                free_cash_flow = current_ebitda * fcf_conversion  # User-defined FCF conversion
                dividend = free_cash_flow * dividend_ratio * ownership
                
                net_cf = dividend - mgmt_fee
                
                if year_end < exit_date:
                    schedule.append({
                        "Company Name": name,
                        "Currency": currency,
                        "Date": year_end,
                        "Type": "Annual Valuation",
                        "EBITDA": round(current_ebitda, 2),
                        "EV": round(current_ev, 2),
                        "Debt": round(debt_balance, 2),
                        "Equity Value": round(current_equity_value, 2),
                        "Investor Equity Cashflow": round(dividend, 2),
                        "Transaction Costs": 0,
                        "Management Fees": round(-mgmt_fee, 2),
                        "Interest Expense": round(-annual_interest, 2),
                        "Debt Paydown": round(-actual_paydown, 2),
                        "Net Cashflow": round(net_cf, 2),
                    })

            # Exit transaction
            years_from_entry = (exit_date - entry_date).days / 365.25
            exit_ebitda = entry_ebitda * ((1 + ebitda_growth) ** years_from_entry)
            exit_ev = exit_ebitda * exit_multiple
            
            # Calculate accrued interest from last annual period to exit
            # Find the last annual period date
            last_annual_year = current_year
            if current_year < exit_year:
                last_annual_year = exit_year - 1
            last_annual_date = date(last_annual_year, 12, 31)
            if last_annual_date >= exit_date:
                # If exit is in the same year as entry, use entry date as last period
                last_annual_date = entry_date
            
            # Calculate interest accrued from last period to exit
            days_since_last_period = (exit_date - last_annual_date).days
            accrued_interest = debt_balance * interest_rate * (days_since_last_period / 365.25)
            
            # Total debt payment includes principal plus accrued interest
            total_debt_payment = debt_balance + accrued_interest
            
            exit_equity_value = exit_ev - total_debt_payment
            exit_proceeds = exit_equity_value * ownership
            exit_txn_costs = exit_ev * txn_costs_pct
            
            final_mgmt_fee = exit_equity_value * mgmt_fee_pct * (years_from_entry - int(years_from_entry))  # Partial year
            
            net_exit_proceeds = exit_proceeds - exit_txn_costs - final_mgmt_fee
            
            schedule.append({
                "Company Name": name,
                "Currency": currency,
                "Date": exit_date,
                "Type": "Exit Sale",
                "EBITDA": round(exit_ebitda, 2),
                "EV": round(exit_ev, 2),
                "Debt": 0,
                "Equity Value": round(exit_equity_value, 2),
                "Investor Equity Cashflow": round(exit_proceeds, 2),
                "Transaction Costs": round(-exit_txn_costs, 2),
                "Management Fees": round(-final_mgmt_fee, 2),
                "Interest Expense": round(-accrued_interest, 2),
                "Debt Paydown": round(-total_debt_payment, 2),
                "Net Cashflow": round(net_exit_proceeds, 2),
            })

            return pd.DataFrame(schedule)

        if st.button("Generate Private Equity Forecast"):
            try:
                schedules = []
                for _, row in pe_df.iterrows():
                    schedules.append(generate_pe_schedule(row))
                full_schedule = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()

                if full_schedule.empty:
                    st.warning("No schedule generated. Please check inputs.")
                else:
                    st.subheader("Projected Cashflows & Valuations")
                    st.dataframe(full_schedule, width='stretch')

                    # Summary
                    st.subheader("Summary by Company")
                    summary = full_schedule.groupby(["Company Name", "Currency"], as_index=False)[
                        ["Investor Equity Cashflow", "Transaction Costs", "Management Fees", "Net Cashflow"]
                    ].sum()
                    
                    # Add returns calculation
                    for idx, row_sum in summary.iterrows():
                        company = row_sum["Company Name"]
                        comp_schedule = full_schedule[full_schedule["Company Name"] == company]
                        dates = pd.to_datetime(comp_schedule["Date"])
                        cashflows = comp_schedule["Net Cashflow"]
                        try:
                            irr = pyxirr.xirr(dates, cashflows)
                            summary.at[idx, "IRR %"] = round(irr * 100, 2)
                            
                            invested = abs(cashflows[cashflows < 0].sum())
                            returned = cashflows[cashflows > 0].sum()
                            moic = returned / invested if invested > 0 else 0
                            summary.at[idx, "MOIC"] = round(moic, 2)
                        except:
                            summary.at[idx, "IRR %"] = None
                            summary.at[idx, "MOIC"] = None
                    
                    # Format MOIC column to display with 2 decimal places
                    if "MOIC" in summary.columns:
                        summary["MOIC"] = summary["MOIC"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                    st.dataframe(summary, width='stretch')

                    # Download
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                        full_schedule.to_excel(writer, sheet_name="Schedule", index=False)
                        summary.to_excel(writer, sheet_name="Summary", index=False)
                    out.seek(0)
                    st.download_button(
                        label="Download Private Equity Forecast as Excel",
                        data=out,
                        file_name="private_equity_forecast.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as ex:
                st.error(f"Error generating private equity forecast: {ex}")

    # --------------------------------------------
    # Infrastructure Cashflow Forecasting
    # --------------------------------------------
    if asset_class == "Infrastructure":
        st.caption("Model long-duration infrastructure assets with contracted revenues, indexation, and lifecycle CapEx.")

        ASSET_TYPES = ["Renewable Energy", "Transport (Toll Road)", "Transport (Airport)", "Utilities", "Social Infrastructure", "Telecom", "Other"]
        REVENUE_MODELS = ["Contracted/Availability", "Regulated (RAB)", "Merchant/Volume"]
        INDEXATION = ["CPI", "RPI", "Fixed %", "None"]
        
        default_infra = pd.DataFrame([
            {
                "Asset Name": "Solar Farm A",
                "Asset Type": "Renewable Energy",
                "Currency": "USD",
                "Investment Amount": 100_000_000.0,
                "Investment Date": pd.to_datetime(date.today() - timedelta(days=365)),
                "Exit Date": pd.to_datetime(date.today() + timedelta(days=9125)),  # 25 years
                "Revenue Model": "Contracted/Availability",
                "Base Annual Revenue": 12_000_000.0,
                "Indexation Type": "CPI",
                "Indexation Rate %": 2.5,
                "Volume Growth % (if Merchant)": 0.0,
                "Fixed OpEx (Annual)": 2_000_000.0,
                "Variable OpEx per Unit": 0.0,
                "OpEx Inflation %": 2.0,
                "Major Maintenance CapEx": 5_000_000.0,
                "Maintenance Frequency (Years)": 5,
                "Leverage %": 70.0,
                "Interest Rate %": 4.5,
                "Amortization Period (Years)": 20,
                "Distribution Payout %": 90.0,
                "Exit EV/Revenue Multiple": 10.0,
            }
        ])

        st.subheader("Infrastructure Inputs")
        st.caption("Edit cells below. Add/remove rows as needed.")
        infra_df = st.data_editor(
            default_infra,
            num_rows="dynamic",
            width='stretch',
            column_config={
                "Asset Type": st.column_config.SelectboxColumn(options=ASSET_TYPES),
                "Revenue Model": st.column_config.SelectboxColumn(options=REVENUE_MODELS),
                "Indexation Type": st.column_config.SelectboxColumn(options=INDEXATION),
                "Investment Amount": st.column_config.NumberColumn(),
                "Base Annual Revenue": st.column_config.NumberColumn(),
                "Indexation Rate %": st.column_config.NumberColumn(format="%.2f"),
                "Volume Growth % (if Merchant)": st.column_config.NumberColumn(format="%.2f"),
                "Fixed OpEx (Annual)": st.column_config.NumberColumn(),
                "Variable OpEx per Unit": st.column_config.NumberColumn(format="$%.2f"),
                "OpEx Inflation %": st.column_config.NumberColumn(format="%.2f"),
                "Major Maintenance CapEx": st.column_config.NumberColumn(),
                "Maintenance Frequency (Years)": st.column_config.NumberColumn(format="%.0f"),
                "Leverage %": st.column_config.NumberColumn(format="%.1f"),
                "Interest Rate %": st.column_config.NumberColumn(format="%.2f"),
                "Amortization Period (Years)": st.column_config.NumberColumn(format="%.0f"),
                "Distribution Payout %": st.column_config.NumberColumn(format="%.1f"),
                "Exit EV/Revenue Multiple": st.column_config.NumberColumn(format="%.1fx"),
                "Investment Date": st.column_config.DateColumn(),
                "Exit Date": st.column_config.DateColumn(),
            },
        )

        def generate_infra_schedule(row: pd.Series) -> pd.DataFrame:
            name = str(row.get("Asset Name", "Asset"))
            currency = str(row.get("Currency", "USD"))
            investment = float(row.get("Investment Amount", 0))
            inv_date = pd.to_datetime(row.get("Investment Date")).date()
            exit_date = pd.to_datetime(row.get("Exit Date")).date()
            base_revenue = float(row.get("Base Annual Revenue", 0))
            indexation_rate = float(row.get("Indexation Rate %", 0)) / 100.0
            volume_growth = float(row.get("Volume Growth % (if Merchant)", 0)) / 100.0
            fixed_opex = float(row.get("Fixed OpEx (Annual)", 0))
            opex_inflation = float(row.get("OpEx Inflation %", 0)) / 100.0
            major_capex = float(row.get("Major Maintenance CapEx", 0))
            capex_freq = float(row.get("Maintenance Frequency (Years)", 5))
            leverage_pct = float(row.get("Leverage %", 0)) / 100.0
            interest_rate = float(row.get("Interest Rate %", 0)) / 100.0
            amort_years = float(row.get("Amortization Period (Years)", 20))
            payout_pct = float(row.get("Distribution Payout %", 0)) / 100.0
            exit_multiple = float(row.get("Exit EV/Revenue Multiple", 10))
            revenue_model = str(row.get("Revenue Model", "Contracted/Availability"))

            schedule = []
            initial_debt = investment * leverage_pct
            debt_balance = initial_debt
            
            # Entry transaction
            equity_investment = investment - initial_debt
            schedule.append({
                "Asset Name": name,
                "Currency": currency,
                "Date": inv_date,
                "Type": "Initial Investment",
                "Revenue": 0,
                "Operating Expenses": 0,
                "Maintenance CapEx": 0,
                "EBITDA": 0,
                "Interest Expense": 0,
                "Principal Payment": 0,
                "Distributable Cash": 0,
                "Distribution to Equity": round(-equity_investment, 2),
                "Debt Balance": round(debt_balance, 2),
                "Net Cashflow": round(-equity_investment, 2),
            })

            # Generate annual periods
            current_year = inv_date.year + 1
            exit_year = exit_date.year
            
            for year in range(current_year, exit_year + 1):
                year_end = date(year, 12, 31)
                if year_end > exit_date:
                    year_end = exit_date
                    
                years_from_inv = (year_end - inv_date).days / 365.25
                
                # Revenue calculation based on model
                if revenue_model == "Merchant/Volume":
                    annual_revenue = base_revenue * ((1 + volume_growth) ** years_from_inv) * ((1 + indexation_rate) ** years_from_inv)
                else:  # Contracted or Regulated
                    annual_revenue = base_revenue * ((1 + indexation_rate) ** years_from_inv)
                
                # Operating expenses
                annual_opex = fixed_opex * ((1 + opex_inflation) ** years_from_inv)
                
                # Major maintenance CapEx (scheduled events)
                if capex_freq > 0 and years_from_inv % capex_freq < 1.0 and years_from_inv >= capex_freq:
                    annual_capex = major_capex
                else:
                    annual_capex = 0
                
                # EBITDA
                ebitda = annual_revenue - annual_opex
                
                # Debt service
                annual_interest = debt_balance * interest_rate
                
                # Amortization schedule
                if amort_years > 0:
                    remaining_periods = max(0, amort_years - years_from_inv)
                    if remaining_periods > 0 and debt_balance > 0:
                        annual_payment = initial_debt * (interest_rate * (1 + interest_rate)**amort_years) / ((1 + interest_rate)**amort_years - 1)
                        principal_payment = min(annual_payment - annual_interest, debt_balance)
                        debt_balance -= principal_payment
                    else:
                        principal_payment = debt_balance
                        debt_balance = 0
                else:
                    principal_payment = 0
                
                # Distributable cash
                distributable = ebitda - annual_interest - principal_payment - annual_capex
                distribution = distributable * payout_pct
                
                if year_end < exit_date:
                    schedule.append({
                        "Asset Name": name,
                        "Currency": currency,
                        "Date": year_end,
                        "Type": "Operating Year",
                        "Revenue": round(annual_revenue, 2),
                        "Operating Expenses": round(-annual_opex, 2),
                        "Maintenance CapEx": round(-annual_capex, 2),
                        "EBITDA": round(ebitda, 2),
                        "Interest Expense": round(annual_interest, 2),
                        "Principal Payment": round(principal_payment, 2),
                        "Distributable Cash": round(distributable, 2),
                        "Distribution to Equity": round(distribution, 2),
                        "Debt Balance": round(debt_balance, 2),
                        "Net Cashflow": round(distribution, 2),
                    })

            # Exit transaction
            years_at_exit = (exit_date - inv_date).days / 365.25
            exit_revenue = base_revenue * ((1 + indexation_rate) ** years_at_exit)
            if revenue_model == "Merchant/Volume":
                exit_revenue *= ((1 + volume_growth) ** years_at_exit)
            
            # Calculate exit operating expenses consistently with operating years
            exit_opex = fixed_opex * ((1 + opex_inflation) ** years_at_exit)
            exit_ebitda = exit_revenue - exit_opex
            
            exit_ev = exit_revenue * exit_multiple  # Infrastructure uses EV/Revenue multiples
            
            # Calculate accrued interest from last operating year to exit
            # Find the last operating year date
            last_operating_year = current_year if current_year < exit_year else exit_year - 1
            last_operating_date = date(last_operating_year, 12, 31)
            if last_operating_date >= exit_date:
                # If exit is in the same year as last operating year, use last operating date
                last_operating_date = date(last_operating_year, 12, 31)
            
            # Calculate interest accrued from last period to exit
            days_since_last_period = (exit_date - last_operating_date).days
            accrued_interest = debt_balance * interest_rate * (days_since_last_period / 365.25)
            
            # Total debt payment includes principal plus accrued interest
            total_debt_payment = debt_balance + accrued_interest
            exit_proceeds = exit_ev - total_debt_payment
            
            schedule.append({
                "Asset Name": name,
                "Currency": currency,
                "Date": exit_date,
                "Type": "Exit Sale",
                "Revenue": round(exit_revenue, 2),
                "Operating Expenses": round(-exit_opex, 2),
                "Maintenance CapEx": 0,
                "EBITDA": round(exit_ebitda, 2),
                "Interest Expense": round(accrued_interest, 2),
                "Principal Payment": round(total_debt_payment, 2),
                "Distributable Cash": round(exit_proceeds, 2),
                "Distribution to Equity": round(exit_proceeds, 2),
                "Debt Balance": 0,
                "Net Cashflow": round(exit_proceeds, 2),
            })

            return pd.DataFrame(schedule)

        if st.button("Generate Infrastructure Forecast"):
            try:
                schedules = []
                for _, row in infra_df.iterrows():
                    schedules.append(generate_infra_schedule(row))
                full_schedule = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()

                if full_schedule.empty:
                    st.warning("No schedule generated. Please check inputs.")
                else:
                    st.subheader("Projected Cashflows")
                    st.dataframe(full_schedule, width='stretch')

                    # Summary
                    st.subheader("Summary by Asset")
                    summary = full_schedule.groupby(["Asset Name", "Currency"], as_index=False)[
                        ["Revenue", "Operating Expenses", "Maintenance CapEx", "EBITDA", 
                         "Interest Expense", "Principal Payment", "Distribution to Equity", "Net Cashflow"]
                    ].sum()
                    
                    # Add returns calculation
                    for idx, row_sum in summary.iterrows():
                        asset = row_sum["Asset Name"]
                        asset_schedule = full_schedule[full_schedule["Asset Name"] == asset]
                        dates = pd.to_datetime(asset_schedule["Date"])
                        cashflows = asset_schedule["Distribution to Equity"]
                        try:
                            irr = pyxirr.xirr(dates, cashflows)
                            summary.at[idx, "IRR %"] = round(irr * 100, 2)
                            
                            invested = abs(cashflows[cashflows < 0].sum())
                            returned = cashflows[cashflows > 0].sum()
                            moic = returned / invested if invested > 0 else 0
                            summary.at[idx, "MOIC"] = round(moic, 2)
                        except:
                            summary.at[idx, "IRR %"] = None
                            summary.at[idx, "MOIC"] = None
                    
                    # Format MOIC column to display with 2 decimal places
                    if "MOIC" in summary.columns:
                        summary["MOIC"] = summary["MOIC"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                    st.dataframe(summary, width='stretch')

                    # Download
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                        full_schedule.to_excel(writer, sheet_name="Schedule", index=False)
                        summary.to_excel(writer, sheet_name="Summary", index=False)
                    out.seek(0)
                    st.download_button(
                        label="Download Infrastructure Forecast as Excel",
                        data=out,
                        file_name="infrastructure_forecast.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as ex:
                st.error(f"Error generating infrastructure forecast: {ex}")
    # --------------------------------------------
    # Loan Cashflow Forecasting & Modelling
    # --------------------------------------------
    if asset_class == "Private Credit":
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
                    "Principal": 1_000_000.0,
                    "OID %": 2,
                    "Cash Interest % (over base)": 4.0,
                    "PIK %": 2.0,
                    "Base Rate Type": "SOFR",
                    "Base Rate %": 5.0,
                    "Interest Frequency": "Quarterly",
                    "Investment Date": pd.to_datetime(date.today() - timedelta(days=45)),
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
            width='stretch',
            column_config={
                "OID %": st.column_config.NumberColumn(format="%.2f"),
                "Cash Interest % (over base)": st.column_config.NumberColumn(format="%.2f"),
                "PIK %": st.column_config.NumberColumn(format="%.2f"),
                "Base Rate %": st.column_config.NumberColumn(format="%.2f"),
                "Exit Price % of Par": st.column_config.NumberColumn(format="%.2f"),
                "Principal": st.column_config.NumberColumn(),
                "Investment Date": st.column_config.DateColumn(),
                "Exit Date": st.column_config.DateColumn(),
                "Interest Frequency": st.column_config.SelectboxColumn(options=list(FREQUENCY_TO_MONTHS.keys())),
                "Base Rate Type": st.column_config.SelectboxColumn(options=["SOFR", "SONIA", "EURIBOR"]),
            },
        )


        def month_end(dt: date) -> date:
            # Returns the last day of the month for a given date
            next_month = dt.replace(day=28) + timedelta(days=4)
            return (next_month - timedelta(days=next_month.day)).date()

    

        def generate_schedule_for_loan(loan_row: pd.Series) -> pd.DataFrame:
            name = str(loan_row.get("Loan Name", "Loan"))
            currency = str(loan_row.get("Currency", "USD"))
            balance = float(loan_row.get("Principal", 0.0))
            oid_pct = float(loan_row.get("OID %", 0.0)) / 100.0
            cash_spread = float(loan_row.get("Cash Interest % (over base)", 0.0)) / 100.0
            pik_rate = float(loan_row.get("PIK %", 0.0)) / 100.0
            base_rate_type = str(loan_row.get("Base Rate Type", "SOFR"))
            base_rate = float(loan_row.get("Base Rate %", 0.0)) / 100.0
            freq = str(loan_row.get("Interest Frequency", "Quarterly"))
            months = FREQUENCY_TO_MONTHS.get(freq, 3)
            investment_date = loan_row.get("Investment Date")
            exit_date = loan_row.get("Exit Date")
            exit_price_pct = float(loan_row.get("Exit Price % of Par", 100.0)) / 100.0

            if pd.isna(investment_date):
                investment_date = pd.to_datetime(date.today())
            if pd.isna(exit_date):
                exit_date = pd.to_datetime(date.today() + timedelta(days=365))

            investment_dt = pd.to_datetime(investment_date).date()
            exit_dt = pd.to_datetime(exit_date).date()
            
            # Get Treasury curve if SOFR is selected as base rate type
            treasury_curve = None
            if base_rate_type == "SOFR":
                treasury_curve = get_treasury_curve(investment_dt, exit_dt)
                if treasury_curve.empty:
                    st.warning(f"Could not fetch Treasury data for {name}. Using static base rate of {base_rate*100:.2f}%.")
                else:
                    loan_duration = (exit_dt - investment_dt).days / 365.25
                    st.info(f"Using dynamic Treasury-based rates for {name}. Investment: {investment_dt}, Exit: {exit_dt} ({loan_duration:.1f}y), First 3M forward rate: {treasury_curve['rate'].iloc[0]*100:.2f}%")

            # Build period dates from investment date to exit
            schedule_rows = []
            
            # Initial loan disbursement (with OID discount)
            oid_discount = balance * oid_pct
            initial_outlay = balance - oid_discount
            schedule_rows.append({
                "Loan Name": name,
                "Currency": currency,
                "Date": investment_dt,
                "Type": "Loan Disbursement",
                "Days": 0,
                "Forward Rate %": 0.0000,  # No interest accrual for disbursement
                "Cash Interest": 0.0,
                "PIK Interest": 0.0,
                "Principal": round(-initial_outlay, 2),
                "Balance": round(balance, 2),
                "Net Cashflow": round(-initial_outlay, 2),
            })

            # Generate coupon dates starting from investment date
            coupon_date = add_months(investment_dt, months)
            last_coupon_date = investment_dt
            while coupon_date < exit_dt:
                period_start = last_coupon_date
                period_end = coupon_date
                days = (period_end - period_start).days
                
                # Get the appropriate base rate for this period
                if base_rate_type == "SOFR" and treasury_curve is not None and not treasury_curve.empty:
                    # Get 3M forward rate starting at the period start date
                    current_forward_rate = get_forward_rate_for_period(period_start, treasury_curve)
                    annual_rate_total = current_forward_rate + cash_spread + pik_rate
                    period_cash_interest = balance * (current_forward_rate + cash_spread) * (days / 365.0)
                else:
                    annual_rate_total = base_rate + cash_spread + pik_rate
                    period_cash_interest = balance * (base_rate + cash_spread) * (days / 365.0)
                
                # Simple ACT/365 accrual
                period_interest_total = balance * annual_rate_total * (days / 365.0)
                period_pik_interest = period_interest_total - period_cash_interest
                balance += period_pik_interest  # Capitalize PIK

                # Get the forward rate used for this period
                if base_rate_type == "SOFR" and treasury_curve is not None and not treasury_curve.empty:
                    period_forward_rate = get_forward_rate_for_period(period_start, treasury_curve)
                    period_forward_rate_display = round(period_forward_rate * 100, 4)  # Convert to percentage
                else:
                    period_forward_rate_display = round(base_rate * 100, 4)  # Static rate
                
                schedule_rows.append({
                    "Loan Name": name,
                    "Currency": currency,
                    "Date": period_end,
                    "Type": "Coupon Payment",
                    "Days": days,
                    "Forward Rate %": period_forward_rate_display,
                    "Cash Interest": round(period_cash_interest, 2),
                    "PIK Interest": round(period_pik_interest, 2),
                    "Principal": 0.0,
                    "Balance": round(balance, 2),
                    "Net Cashflow": round(period_cash_interest, 2),
                })

                last_coupon_date = coupon_date
                coupon_date = add_months(coupon_date, months)

            # Final partial period from last coupon to exit
            if last_coupon_date < exit_dt:
                days = (exit_dt - last_coupon_date).days
                
                # Get the appropriate base rate for the final period
                if base_rate_type == "SOFR" and treasury_curve is not None and not treasury_curve.empty:
                    # Get 3M forward rate starting at the last coupon date
                    current_forward_rate = get_forward_rate_for_period(last_coupon_date, treasury_curve)
                    annual_rate_total = current_forward_rate + cash_spread + pik_rate
                    period_cash_interest = balance * (current_forward_rate + cash_spread) * (days / 365.0)
                else:
                    annual_rate_total = base_rate + cash_spread + pik_rate
                    period_cash_interest = balance * (base_rate + cash_spread) * (days / 365.0)
                
                period_interest_total = balance * annual_rate_total * (days / 365.0)
                period_pik_interest = period_interest_total - period_cash_interest
                # Accrued interest: cash portion accrues but is not paid until exit; PIK accrues but is not capitalized until exit

                accrued_cash = period_cash_interest
                accrued_pik = period_pik_interest
                balance_at_exit_before_pik_cap = balance
                balance += accrued_pik  # Capitalize PIK at exit

                # Exit flows
                principal_repaid = balance_at_exit_before_pik_cap * exit_price_pct
                premium_or_discount = principal_repaid - balance_at_exit_before_pik_cap

                # Get the forward rate used for the final period
                if base_rate_type == "SOFR" and treasury_curve is not None and not treasury_curve.empty:
                    final_forward_rate = get_forward_rate_for_period(last_coupon_date, treasury_curve)
                    final_forward_rate_display = round(final_forward_rate * 100, 4)
                else:
                    final_forward_rate_display = round(base_rate * 100, 4)
                
                schedule_rows.append({
                    "Loan Name": name,
                    "Currency": currency,
                    "Date": exit_dt,
                    "Type": "Exit Accrual",
                    "Days": days,
                    "Forward Rate %": final_forward_rate_display,
                    "Cash Interest": round(accrued_cash, 2),
                    "PIK Interest": round(accrued_pik, 2),
                    "Principal": 0.0,
                    "Balance": round(balance, 2),
                    "Net Cashflow": round(accrued_cash, 2),
                })

                schedule_rows.append({
                    "Loan Name": name,
                    "Currency": currency,
                    "Date": exit_dt,
                    "Type": "Exit Payment",
                    "Days": 0,
                    "Forward Rate %": 0.0000,  # No interest accrual for principal payment
                    "Cash Interest": 0.0,
                    "PIK Interest": 0.0,
                    "Principal": round(principal_repaid, 2),
                    "Balance": round(balance - principal_repaid, 2),
                    "Net Cashflow": round(principal_repaid, 2),
                })

            schedule_df = pd.DataFrame(schedule_rows)
            if not schedule_df.empty:
                schedule_df = schedule_df.sort_values(["Loan Name", "Date"]).reset_index(drop=True)
            return schedule_df


        if st.button("Generate Loan Cashflow Forecast"):
            try:
                # Check if any loans use SOFR and show Treasury market data
                sofr_loans = loans_df[loans_df.get("Base Rate Type", "") == "SOFR"]
                if not sofr_loans.empty:
                    st.info("Fetching Treasury Constant Maturity data from FRED API...")
                    
                    # Get the Treasury rates used for bootstrapping
                    treasury_series = {
                        '3M': 'DGS3MO', '6M': 'DGS6MO', '1Y': 'DGS1', 
                        '2Y': 'DGS2', '5Y': 'DGS5', '10Y': 'DGS10', '30Y': 'DGS30'
                    }
                    
                    # Get investment dates from loans to show rate context
                    investment_dates = loans_df['Investment Date'].unique()
                    if len(investment_dates) > 0:
                        sample_investment_date = pd.to_datetime(investment_dates[0]).date()
                        st.info(f"📅 Using Treasury rates as of investment date: {sample_investment_date}")
                    
                    treasury_rates = {}
                    for tenor, series_id in treasury_series.items():
                        # Use the sample investment date for rate display
                        end_date_str = sample_investment_date.strftime('%Y-%m-%d') if len(investment_dates) > 0 else None
                        start_date_str = None
                        if len(investment_dates) > 0:
                            start_date = sample_investment_date - timedelta(days=90)
                            start_date_str = start_date.strftime('%Y-%m-%d')
                        
                        data = fetch_fred_data(series_id, start_date=start_date_str, end_date=end_date_str)
                        if not data.empty:
                            investment_date_pd = pd.to_datetime(sample_investment_date) if len(investment_dates) > 0 else pd.Timestamp.now()
                            data_filtered = data[data['date'] <= investment_date_pd]
                            if not data_filtered.empty:
                                treasury_rates[tenor] = data_filtered['value'].iloc[-1]
                    
                    if treasury_rates:
                        st.success(f"✅ Treasury market data loaded successfully!")
                        rate_display = ", ".join([f"{tenor} = {rate:.2f}%" for tenor, rate in treasury_rates.items()])
                        st.info(f"📊 Market Data: {rate_display}")
                        
                        # Show methodology
                        st.caption("💡 **Treasury-Based Discount Factor Bootstrapping:**")
                        st.caption("• **Formula**: f(Tk, Tk+1) = D(Tk) / D(Tk+1) - 1")
                        st.caption("• **Discount Factors**: D(T) = 1 / (1 + R(T) × T)")
                        st.caption("• **Using Treasury Constant Maturity Yields as base rates**")
                        st.caption("• **Forward curves are anchored to each loan's investment date**")
                        st.caption("• Each SOFR loan will use Treasury-based forward rates")
                    else:
                        st.warning("⚠️ Could not fetch Treasury data. Using static base rates for SOFR loans.")
                
                schedules = []
                for _, row in loans_df.iterrows():
                    schedules.append(generate_schedule_for_loan(row))
                full_schedule = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()

                if full_schedule.empty:
                    st.warning("No schedule generated. Please check inputs.")
                else:
                    st.subheader("Projected Cashflows")
                    st.dataframe(full_schedule, width='stretch')

                    # Summary per loan
                    st.subheader("Summary by Loan")
                    summary = (
                        full_schedule.groupby(["Loan Name", "Currency"], as_index=False)[["Forward Rate %", "Cash Interest", "PIK Interest", "Principal", "Net Cashflow"]]
                        .agg({
                            "Forward Rate %": "mean",  # Average forward rate
                            "Cash Interest": "sum",
                            "PIK Interest": "sum", 
                            "Principal": "sum",
                            "Net Cashflow": "sum"
                        })
                    )
                    
                    # Add returns calculation
                    for idx, row_sum in summary.iterrows():
                        loan_name = row_sum["Loan Name"]
                        loan_schedule = full_schedule[full_schedule["Loan Name"] == loan_name]
                        dates = pd.to_datetime(loan_schedule["Date"])
                        cashflows = loan_schedule["Net Cashflow"]
                        try:
                            irr = pyxirr.xirr(dates, cashflows)
                            summary.at[idx, "IRR %"] = round(irr * 100, 2)
                            
                            invested = abs(cashflows[cashflows < 0].sum())
                            returned = cashflows[cashflows > 0].sum()
                            moic = returned / invested if invested > 0 else 0
                            summary.at[idx, "MOIC"] = round(moic, 2)
                        except:
                            summary.at[idx, "IRR %"] = None
                            summary.at[idx, "MOIC"] = None
                    
                    # Format MOIC column to display with 2 decimal places
                    if "MOIC" in summary.columns:
                        summary["MOIC"] = summary["MOIC"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                    st.dataframe(summary, width='stretch')

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
