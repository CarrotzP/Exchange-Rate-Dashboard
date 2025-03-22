import pandas as pd
from freecurrencyapi import Client
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import numpy as np

# ╔══════════════════════════════════════╗
# ║           IMPORTS SECTION            ║
# ╚══════════════════════════════════════╝
# - pandas: สำหรับการจัดการและวิเคราะห์ข้อมูลในรูปแบบ DataFrame
# - freecurrencyapi: ใช้ดึงข้อมูลอัตราแลกเปลี่ยนจาก API
# - datetime: จัดการข้อมูลวันที่และเวลา
# - plotly.express & plotly.graph_objects: สร้างกราฟแบบโต้ตอบได้
# - streamlit: ใช้สร้าง Dashboard แบบโต้ตอบ
# - os: จัดการไฟล์และพาธในระบบ
# - numpy: ใช้สำหรับการคำนวณทางคณิตศาสตร์

# ╔══════════════════════════════════════╗
# ║           COLOR SETTINGS             ║
# ╚══════════════════════════════════════╝
# กำหนดสีโทนพาสเทลสำหรับกราฟ เพื่อให้ดูสบายตา
pastel_colors = ['#FFCCCB', '#ADD8E6', '#90EE90', '#D8BFD8', '#F0E68C', 
                 '#FFDAB9', '#E6E6FA', '#B0E0E6', '#FFFACD', '#D8BFD8']

# ╔══════════════════════════════════════╗
# ║         DATA WAREHOUSE CREATION      ║
# ╚══════════════════════════════════════╝
# โหลดข้อมูลจากไฟล์ CSV และ API เพื่อสร้าง Data Warehouse
# รวมข้อมูลอัตราแลกเปลี่ยนตั้งแต่ปี 2000-2025
csv_file = "exchange_rate_data.csv"  # ไฟล์ข้อมูลอัตราแลกเปลี่ยน
if not os.path.exists(csv_file):
    st.error(f"Error: File '{csv_file}' not found. Please ensure the file exists in the correct directory.")
    st.stop()

try:
    df_csv = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Error reading CSV file: {str(e)}")
    st.stop()

selected_columns = ["Date", "Australian Dollar", "Euro", "Japanese Yen", "Thai Baht", "U.K. Pound Sterling", "U.S. Dollar"]
if not all(col in df_csv.columns for col in selected_columns):
    st.error(f"Error: CSV file must contain the following columns: {selected_columns}")
    st.stop()
df_csv = df_csv[selected_columns]
df_csv["Date"] = pd.to_datetime(df_csv["Date"], errors='coerce')

# ดึงข้อมูลรายปี (มกราคม) จาก CSV สำหรับปี 2000-2018
df_csv_yearly = pd.DataFrame()
for year in range(2000, 2019):
    jan_data = df_csv[(df_csv["Date"].dt.year == year) & (df_csv["Date"].dt.month == 1)]
    valid_row = jan_data.dropna(subset=["Australian Dollar", "Euro", "Japanese Yen", "Thai Baht", "U.K. Pound Sterling"]).iloc[0:1]
    if not valid_row.empty:
        df_csv_yearly = pd.concat([df_csv_yearly, valid_row])

# ดึงข้อมูลจาก API สำหรับปี 2019-2025
API_KEY = "fca_live_dHv55hoVcCdeyS8GFrG7adng5NG4wHKxxCFflRSW"
client = Client(API_KEY)
currencies_api = ["AUD", "EUR", "JPY", "THB", "GBP"]  # สกุลเงินที่ใช้ใน API

def fetch_yearly_rates_complete(start_year=2019, end_year=2025):
    data = {}
    for year in range(start_year, end_year + 1):
        found = False
        for day in range(2, 32):
            date_str = f"{year}-01-{day:02d}"
            try:
                result = client.historical(date=date_str, base_currency="USD", currencies=currencies_api)
                rates = result["data"][date_str]
                if all(rates.get(currency) is not None for currency in currencies_api):
                    data[date_str] = rates
                    found = True
                    break
            except Exception as e:
                print(f"ไม่สามารถดึงข้อมูลสำหรับวันที่ {date_str}: {str(e)}")
        if not found:
            print(f"ไม่พบวันที่มีข้อมูลครบในปี {year}")
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    return df

api_file = "api_data_2019_2025.csv"
if os.path.exists(api_file):
    print(f"โหลดข้อมูลจากไฟล์ {api_file}...")
    df_api_yearly = pd.read_csv(api_file)
    df_api_yearly["Date"] = pd.to_datetime(df_api_yearly["Date"], errors='coerce')
else:
    print("ดึงข้อมูลจาก API (2019-2025)...")
    try:
        df_api_yearly = fetch_yearly_rates_complete()
        currency_name_mapping = {"AUD": "Australian Dollar", "EUR": "Euro", "JPY": "Japanese Yen", "THB": "Thai Baht", "GBP": "U.K. Pound Sterling"}
        df_api_yearly = df_api_yearly.rename(columns=currency_name_mapping)
        df_api_yearly["U.S. Dollar"] = 1.0
        df_api_yearly.reset_index(inplace=True)
        df_api_yearly.rename(columns={"index": "Date"}, inplace=True)
        df_api_yearly.to_csv(api_file, index=False)
        print(f"บันทึกข้อมูล API เป็น {api_file}")
    except Exception as e:
        st.error(f"Error fetching API data: {str(e)}")
        st.stop()

# รวมข้อมูลจาก CSV และ API แล้วบันทึกเป็น Data Warehouse
df_warehouse = pd.concat([df_csv_yearly, df_api_yearly], ignore_index=True).sort_values("Date")
df_warehouse.to_csv("exchange_rate_warehouse_yearly_complete_2000_2025.csv", index=False)
print("สร้าง Data Warehouse เสร็จสิ้น!")
print(df_warehouse)

# ╔══════════════════════════════════════╗
# ║           DATA ANALYSIS              ║
# ╚══════════════════════════════════════╝
# วิเคราะห์ข้อมูลอัตราแลกเปลี่ยนและคำนวณสถิติพื้นฐาน
df = pd.read_csv("exchange_rate_warehouse_yearly_complete_2000_2025.csv")
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
currencies = ["Australian Dollar", "Euro", "Japanese Yen", "Thai Baht", "U.K. Pound Sterling"]
df["Date_numeric"] = df["Date"].map(lambda x: x.timestamp())

# คำนวณสถิติสำหรับแต่ละสกุลเงิน (Mean, Max Change, Min Change, Std Dev, Median)
stats = {}
for currency in currencies:
    stats[currency] = {
        "Mean": df[currency].mean(),
        "Max Change (%)": ((df[currency].pct_change().max()) * 100).round(2) if not df[currency].empty else 0,
        "Min Change (%)": ((df[currency].pct_change().min()) * 100).round(2) if not df[currency].empty else 0,
        "Std Dev": df[currency].std(),
        "Median": df[currency].median()
    }
stats_df = pd.DataFrame(stats).T
print("\nสถิติ 5 รายการ:")
print(stats_df)

# ╔══════════════════════════════════════╗
# ║         PLOTLY GRAPH CREATION        ║
# ╚══════════════════════════════════════╝
# สร้างกราฟ 10 รายการด้วย Plotly เพื่อแสดงผลใน Dashboard
# กราฟ 1: Bar Yearly % Change
pct_change_df = df[currencies].pct_change().dropna() * 100
dates_for_pct = df["Date"][1:]
fig1 = go.Figure()
for i, currency in enumerate(currencies):
    fig1.add_trace(go.Bar(x=dates_for_pct, y=pct_change_df[currency], name=currency, marker_color=pastel_colors[i], opacity=0.7))
fig1.update_layout(title="Yearly % Change of All Currencies", xaxis_title="Year", yaxis_title="% Change", 
                   legend_title="Currencies", showlegend=True, barmode='group')
fig1.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

# กราฟ 2: THB with MA
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["Date"], y=df["Thai Baht"], name="Thai Baht", line=dict(color=pastel_colors[3])))
ma5 = df["Thai Baht"].rolling(window=5, center=True).mean()
fig2.add_trace(go.Scatter(x=df["Date"], y=ma5, name="5-Year MA", line=dict(dash="dot", color=pastel_colors[1])))
ma10 = df["Thai Baht"].rolling(window=10, center=True).mean()
fig2.add_trace(go.Scatter(x=df["Date"], y=ma10, name="10-Year MA", line=dict(dash="dash", color=pastel_colors[4])))
fig2.update_layout(title="Thai Baht with Moving Averages", xaxis_title="Year", yaxis_title="Exchange Rate (THB/USD)", 
                   legend_title="THB Metrics")

# กราฟ 3: Scatter EUR vs GBP
fig3 = px.scatter(df, x="Euro", y="U.K. Pound Sterling", color=df["Date"].dt.year.astype(str), 
                  color_discrete_sequence=pastel_colors, labels={"color": "Year"}, hover_data=["Date"], 
                  size_max=10)
fig3.update_traces(marker=dict(size=12))
coeff = np.polyfit(df["Euro"], df["U.K. Pound Sterling"], 2)
x = np.linspace(df["Euro"].min(), df["Euro"].max(), 100)
trend = np.polyval(coeff, x)
fig3.add_trace(go.Scatter(x=x, y=trend, name="Quadratic Trend", line=dict(dash="dash", color=pastel_colors[2])))
fig3.update_layout(title="Euro vs U.K. Pound Sterling Relationship", xaxis_title="Euro (vs USD)", yaxis_title="U.K. Pound Sterling (vs USD)")

# กราฟ 4: Correlation Heatmap
fig4 = px.imshow(df[currencies].corr(), text_auto=".2f", color_continuous_scale="Blues", 
                 title="Correlation Between Currencies",
                 labels=dict(x="Currency", y="Currency", color="Correlation"))
fig4.update_layout(width=600, height=600)
fig4.update_xaxes(ticktext=currencies, tickvals=currencies)
fig4.update_yaxes(ticktext=currencies, tickvals=currencies)

# กราฟ 5: Area Chart JPY
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df["Date"], y=df["Japanese Yen"], fill="tozeroy", name="Japanese Yen Rate (Area)", 
                          line_color=pastel_colors[5], fillcolor=pastel_colors[2], opacity=0.5))
fig5.add_trace(go.Scatter(x=df["Date"], y=df["Japanese Yen"], name="Japanese Yen Rate (Line)", line_color=pastel_colors[6]))
fig5.update_layout(title="Japanese Yen Exchange Rate (Area Chart)", xaxis_title="Year", yaxis_title="Exchange Rate (JPY/USD)", 
                   legend_title="JPY Metrics")

# กราฟ 6: Box Plot AUD, EUR, GBP
subset_currencies = ["Australian Dollar", "Euro", "U.K. Pound Sterling"]
fig6 = go.Figure()
for i, currency in enumerate(subset_currencies):
    fig6.add_trace(go.Box(y=df[currency], name=currency, marker_color=pastel_colors[i], boxmean=True))
fig6.update_layout(title="Exchange Rate Distribution (Australian Dollar, Euro, U.K. Pound Sterling)", xaxis_title="Currency", yaxis_title="Exchange Rate (vs USD)")

# กราฟ 7: Donut Chart THB ทุก 5 ปี
df_thb_5year = df[df["Date"].dt.year % 5 == 0]
fig7 = px.pie(df_thb_5year, values="Thai Baht", names=df_thb_5year["Date"].dt.year, hole=0.3, 
              color_discrete_sequence=pastel_colors[:len(df_thb_5year)], 
              title="Thai Baht Exchange Rate Distribution Every 5 Years (Donut Chart)")
fig7.update_traces(textinfo='percent+label', hoverinfo='label+percent+value')

# กราฟ 8: GBP Candlestick
df_candle = pd.DataFrame({
    "Date": df["Date"],
    "Open": df["U.K. Pound Sterling"].shift(1),
    "High": df["U.K. Pound Sterling"].rolling(window=2).max(),
    "Low": df["U.K. Pound Sterling"].rolling(window=2).min(),
    "Close": df["U.K. Pound Sterling"]
}).dropna()
fig8 = go.Figure(data=[go.Candlestick(x=df_candle["Date"], open=df_candle["Open"], high=df_candle["High"], 
                                      low=df_candle["Low"], close=df_candle["Close"], 
                                      increasing_line_color=pastel_colors[2], decreasing_line_color=pastel_colors[3])])
fig8.update_layout(title="U.K. Pound Sterling Candlestick Chart", xaxis_title="Year", yaxis_title="Exchange Rate (GBP/USD)")

# กราฟ 9: Stacked Bar Max/Min % Change
fig9 = go.Figure()
fig9.add_trace(go.Bar(x=stats_df.index, y=stats_df["Max Change (%)"], name="Max % Change", marker_color=pastel_colors[2]))
fig9.add_trace(go.Bar(x=stats_df.index, y=stats_df["Min Change (%)"], name="Min % Change", marker_color=pastel_colors[3]))
fig9.add_trace(go.Bar(x=stats_df.index, y=stats_df["Mean"], name="Mean Rate", marker_color=pastel_colors[4], opacity=0.5))
fig9.update_layout(title="Max, Min % Change & Mean by Currency", xaxis_title="Currency", yaxis_title="% Change / Rate", 
                   barmode='overlay')

# กราฟ 10: Bar JPY % Change รายปี
jpy_pct_change = df["Japanese Yen"].pct_change().dropna() * 100
fig10 = go.Figure()
fig10.add_trace(go.Bar(x=dates_for_pct, y=jpy_pct_change, name="Japanese Yen % Change", marker_color=pastel_colors[2], opacity=0.7))
ma3_jpy = jpy_pct_change.rolling(window=3, center=True).mean()
fig10.add_trace(go.Scatter(x=dates_for_pct, y=ma3_jpy, name="3-Year MA", line=dict(dash="dash", color=pastel_colors[1])))
fig10.update_layout(title="Japanese Yen Yearly % Change", xaxis_title="Year", yaxis_title="% Change", 
                    legend_title="JPY Metrics")
fig10.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

# ╔══════════════════════════════════════╗
# ║         STREAMLIT DASHBOARD          ║
# ╚══════════════════════════════════════╝
# สร้าง Dashboard ด้วย Streamlit เพื่อแสดงข้อมูลและกราฟ
st.set_page_config(page_title="Exchange Rate Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Exchange Rate Dashboard (2000-2025)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Explore yearly exchange rates for Australian Dollar, Euro, Japanese Yen, Thai Baht, and U.K. Pound Sterling against USD</p>", unsafe_allow_html=True)

with st.expander("Data & Statistics", expanded=True):
    # ปรับเลย์เอาต์ให้สมดุลเท่ากัน (1:1)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Warehouse")
        # ปรับแต่งตารางให้สมดุลกับ Key Statistics
        st.dataframe(
            df.drop(columns=["Date_numeric"]).style.set_properties(**{
                'background-color': '#F9FAFB',  # พื้นหลังสีอ่อน
                'color': '#333333',
                'border': '1px solid #E5E7EB',  # เส้นขอบบาง
                'font-size': '14px', 
            }),
            height=250,  # ปรับความสูงให้สมดุลกับ Key Statistics
            use_container_width=True
        )
        # เพิ่มปุ่มดาวน์โหลด
        st.download_button(
            label="Download Data as CSV",
            data=df.drop(columns=["Date_numeric"]).to_csv(index=False),
            file_name="exchange_rate_data.csv",
            mime="text/csv"
        )

    with col2:
        st.subheader("Key Statistics")
        # สร้างการ์ดแนวนอน (ซ้ายไปขวา)
        cols = st.columns(len(stats_df))  # สร้างคอลัมน์ตามจำนวนสกุลเงิน (5 คอลัมน์)
        for idx, (currency, stats) in enumerate(stats_df.iterrows()):
            with cols[idx]:  # จัดเรียงการ์ดในคอลัมน์ที่ idx
                card_color = pastel_colors[idx % len(pastel_colors)]
                # ปรับขนาดและสไตล์การ์ดให้เหมาะสม
                card_html = f"""
                <div style='background-color: {card_color}; padding: 10px; border-radius: 8px; text-align: center;'>
                    <h4 style='color: #333; margin: 0; font-size: 14px;'>{currency}</h4>
                    <p style='color: #555; margin: 3px 0; font-size: 11px;'><b>Mean:</b> {stats['Mean']:.4f}</p>
                    <p style='color: #555; margin: 3px 0; font-size: 11px;'><b>Max Change:</b> {stats['Max Change (%)']:.2f}%</p>
                    <p style='color: #555; margin: 3px 0; font-size: 11px;'><b>Min Change:</b> {stats['Min Change (%)']:.2f}%</p>
                    <p style='color: #555; margin: 3px 0; font-size: 11px;'><b>Std Dev:</b> {stats['Std Dev']:.4f}</p>
                    <p style='color: #555; margin: 3px 0; font-size: 11px;'><b>Median:</b> {stats['Median']:.4f}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

st.subheader("Visualizations")
with st.container():
    st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
with col2:
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)

with st.expander("Additional Charts", expanded=False):
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig5, use_container_width=True)
        st.plotly_chart(fig7, use_container_width=True)
    with col4:
        st.plotly_chart(fig9, use_container_width=True)
        st.plotly_chart(fig10, use_container_width=True)
        st.plotly_chart(fig8, use_container_width=True)

st.markdown(
    "<hr><p style='text-align: center; color: #888;'>"
    "Created by 6630611058 | Data Source: "
    "<a href='https://www.kaggle.com/datasets/thebasss/currency-exchange-rates/data' style='color: #888; text-decoration: underline;'>Kaggle</a> & "
    "<a href='https://freecurrencyapi.com/' style='color: #888; text-decoration: underline;'>freecurrencyapi.com</a>"
    "</p>",
    unsafe_allow_html=True
)

print("Dashboard พร้อมใช้งาน! รันด้วย 'streamlit run your_script.py'")
