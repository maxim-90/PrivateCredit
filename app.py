import streamlit as st
import pandas as pd
import pyxirr

st.title("IRR Calculator")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Define the compute_irr function (same as before)
    def compute_irr(data):
        cashflows = data['Cashflow'].tolist()
        dates = data['Date'].tolist()
        date_strs = [date.strftime('%Y-%m-%d') for date in dates]
        irr = pyxirr.xirr(date_strs, cashflows)
        return irr

    irr_by_deal = df.groupby('Deal').apply(compute_irr)

    st.write("## IRR Results")
    st.dataframe(irr_by_deal)

if __name__ == "__main__":
    st.run(debug=True, host="0.0.0.0", port=5000)