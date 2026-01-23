import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Vietnam Market Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Dark Mode & Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background-color: #1f2937; border: 1px solid #374151;
        padding: 15px; border-radius: 8px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val { font-size: 1.6rem; font-weight: bold; color: #fff; }
    .metric-lbl { font-size: 0.9rem; color: #9ca3af; margin-bottom: 5px; }
    .gain { color: #4ade80; }
    .loss { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---

def clean_currency(val):
    """Làm sạch chuỗi tiền tệ"""
    if pd.isna(val) or val == '':
        return np.nan
    clean_str = str(val).replace('₫', '').replace(',', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return np.nan

@st.cache_data
def load_and_merge_data():
    """Đọc và xử lý file CSV"""
    files = {
        "Giá - Chứng khoán.csv": "Chứng khoán",
        "Giá - Thép & Vật liệu xây dựng.csv": "Thép & VLXD",
        "Giá- Bất động sản.csv": "Bất động sản",
        "Giá- Ngân hàng.csv": "Ngân hàng"
    }
    
    all_data = []

    for filename, sector_name in files.items():
        if not os.path.exists(filename):
            continue
            
        try:
            df_raw = pd.read_csv(filename)
            df_raw.columns = [c.strip() for c in df_raw.columns]
            
            time_col = next((c for c in df_raw.columns if 'time' in c.lower()), None)
            if not time_col:
                continue
            
            df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors='coerce')
            
            close_cols = [c for c in df_raw.columns if 'close' in c.lower()]
            
            for col in close_cols:
                ticker_match = re.split(r'[_ ]', col)[0].upper()
                if len(ticker_match) > 4 or len(ticker_match) < 3: 
                    continue
                    
                temp_df = pd.DataFrame()
                temp_df['Date'] = df_raw[time_col]
                temp_df['Close'] = df_raw[col].apply(clean_currency)
                temp_df['Ticker'] = ticker_match
                temp_df['Sector'] = sector_name
                
                vol_col = next((c for c in df_raw.columns if ticker_match in c.upper() and 'volume' in c.lower()), None)
                if vol_col:
                    temp_df['Volume'] = df_raw[vol_col].apply(clean_currency) 
                else:
                    temp_df['Volume'] = 0
                
                all_data.append(temp_df)
                
        except Exception as e:
            st.error(f"Lỗi file {filename}: {str(e)}")

    if not all_data:
        return pd.DataFrame()
        
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.dropna(subset=['Date', 'Close'])
    df_final = df_final.sort_values(['Ticker', 'Date'])
    
    return df_final

# --- 3. TÍNH TOÁN CHỈ SỐ KỸ THUẬT ---

def calculate_technical_indicators(df):
    df = df.copy()
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands & MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    
    # Drawdown (Từ đỉnh lịch sử)
    df['Peak'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
    
    return df

# --- 4. GIAO DIỆN CHÍNH ---

df = load_and_merge_data()

if df.empty:
    st.error("Không tìm thấy dữ liệu. Hãy kiểm tra tên file CSV trên GitHub.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("BỘ LỌC DỮ LIỆU")
min_date, max_date = df['Date'].min(), df['Date'].max()
start_date = st.sidebar.date_input("Từ ngày", min_date)
end_date = st.sidebar.date_input("Đến ngày", max_date)

sectors = df['Sector'].unique()
selected_sectors = st.sidebar.multiselect("Chọn Ngành", sectors, default=sectors)

df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                 (df['Date'] <= pd.to_datetime(end_date)) & 
                 (df['Sector'].isin(selected_sectors))]

available_tickers = df_filtered['Ticker'].unique()
selected_tickers = st.sidebar.multiselect("Chọn Mã (Max 5)", available_tickers, default=available_tickers[:3])

if not selected_tickers:
    st.warning("Vui lòng chọn ít nhất 1 mã cổ phiếu.")
    st.stop()

df_display = df_filtered[df_filtered['Ticker'].isin(selected_tickers)]

# --- MAIN PAGE ---
st.title("PRO TRADING ANALYTICS")
st.markdown("---")

# KPI Cards
cols = st.columns(len(selected_tickers))
for i, ticker in enumerate(selected_tickers):
    ticker_data = df_display[df_display['Ticker'] == ticker]
    if not ticker_data.empty:
        curr_price = ticker_data.iloc[-1]['Close']
        prev_price = ticker_data.iloc[-2]['Close'] if len(ticker_data) > 1 else curr_price
        change = ((curr_price - prev_price) / prev_price) * 100
        
        color_cls = "gain" if change >= 0 else "loss"
        symbol = "▲" if change >= 0 else "▼"
        
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">{ticker}</div>
                <div class="metric-val">{curr_price:,.0f}</div>
                <div class="{color_cls}">{symbol} {change:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("###")

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["Sức Mạnh Giá", "Phân Tích Kỹ Thuật", "Xu Hướng Mùa Vụ", "Rủi Ro Drawdown"])

with tab1:
    st.markdown("##### So sánh tăng trưởng (%) từ đầu kỳ")
    pivot = df_display.pivot(index='Date', columns='Ticker', values='Close')
    pivot_norm = pivot.apply(lambda x: (x / x.iloc[0] - 1) * 100)
    fig_norm = px.line(pivot_norm, x=pivot_norm.index, y=pivot_norm.columns, template='plotly_dark')
    st.plotly_chart(fig_norm, use_container_width=True)

with tab2:
    target_ticker = st.selectbox("Chọn mã soi chi tiết:", selected_tickers)
    tech_df = df[df['Ticker'] == target_ticker].copy().sort_values('Date')
    tech_df = calculate_technical_indicators(tech_df)
    mask = (tech_df['Date'] >= pd.to_datetime(start_date)) & (tech_df['Date'] <= pd.to_datetime(end_date))
    tech_view = tech_df.loc[mask]
    
    fig_tech = go.Figure()
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Close'], name='Close', line=dict(color='white')))
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['MA20'], name='MA20', line=dict(color='yellow')))
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Upper_Band'], line=dict(color='gray', width=0), showlegend=False))
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Lower_Band'], line=dict(color='gray', width=0), fill='tonexty', showlegend=False))
    st.plotly_chart(fig_tech, use_container_width=True)
    
    fig_rsi = px.line(tech_view, x='Date', y='RSI', title="Chỉ số RSI")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(template='plotly_dark', height=250)
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab3:
    st.markdown("##### Xu hướng Mùa vụ (Theo tháng)")
    
    # Logic: Tính % change từng tháng -> Cộng dồn (Accumulate)
    full_history = df[df['Ticker'].isin(selected_tickers)].copy()
    full_history['Month'] = full_history['Date'].dt.month
    full_history['Year'] = full_history['Date'].dt.year
    
    monthly_close = full_history.groupby(['Ticker', 'Year', 'Month'])['Close'].last().reset_index()
    monthly_close['Pct_Change'] = monthly_close.groupby('Ticker')['Close'].pct_change()
    
    seasonality_avg = monthly_close.groupby(['Ticker', 'Month'])['Pct_Change'].mean().reset_index()
    
    season_chart_data = []
    for ticker in selected_tickers:
        t_data = seasonality_avg[seasonality_avg['Ticker'] == ticker].sort_values('Month')
        # Bắt đầu từ 100
        t_data['Cumulative_Trend'] = (1 + t_data['Pct_Change'].fillna(0)).cumprod() * 100
        
        # Thêm điểm đầu (Tháng 0 = 100)
        start_point = pd.DataFrame({'Ticker': [ticker], 'Month': [0], 'Cumulative_Trend': [100]})
        t_data = pd.concat([start_point, t_data], ignore_index=True)
        season_chart_data.append(t_data)
    
    if season_chart_data:
        df_season = pd.concat(season_chart_data)
        fig_season = px.line(df_season, x='Month', y='Cumulative_Trend', color='Ticker',
                            template='plotly_dark', labels={'Cumulative_Trend': 'Chỉ số (Gốc=100)'})
        fig_season.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        fig_season.add_hline(y=100, line_dash="dash", line_color="white", opacity=0.5)
        st.plotly_chart(fig_season, use_container_width=True)

with tab4:
    st.markdown("##### Drawdown (Sụt giảm từ đỉnh lịch sử)")
    dd_data = pd.DataFrame()
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].sort_values('Date').copy()
        t_df['Peak'] = t_df['Close'].cummax()
        t_df['Drawdown'] = (t_df['Close'] - t_df['Peak']) / t_df['Peak'] * 100
        
        mask_dd = (t_df['Date'] >= pd.to_datetime(start_date)) & (t_df['Date'] <= pd.to_datetime(end_date))
        dd_data[ticker] = t_df.loc[mask_dd].set_index('Date')['Drawdown']
    
    fig_dd = px.area(dd_data, template='plotly_dark')
    st.plotly_chart(fig_dd, use_container_width=True)

# --- 5. BÁO CÁO TỰ ĐỘNG (AUTO REPORT) ---
st.markdown("---")
st.header("Báo Cáo Phân Tích Tự Động")

def generate_insight(ticker, df_input):
    last_row = df_input.iloc[-1]
    
    # 1. Trend
    trend = "TĂNG" if last_row['Close'] > last_row['MA20'] else "GIẢM"
    
    # 2. RSI
    rsi = last_row['RSI']
    rsi_signal = "Trung tính"
    if rsi > 70: rsi_signal = "QUÁ MUA"
    elif rsi < 30: rsi_signal = "QUÁ BÁN"
    
    return f"""
    **Mã: {ticker}** ({last_row['Date'].strftime('%d/%m/%Y')})
    - Giá: {last_row['Close']:,.0f}
    - Xu hướng: {trend} (vs MA20)
    - RSI: {rsi:.1f} ({rsi_signal})
    - Khuyến nghị: {'Chốt lời dần' if rsi > 70 else ('Cân nhắc mua' if rsi < 30 else 'Nắm giữ')}
    """

col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    st.info("Bảng Tóm tắt")
    report_df = pd.DataFrame()
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
        t_df = calculate_technical_indicators(t_df)
        last_row = t_df.iloc[-1]
        report_df = pd.concat([report_df, pd.DataFrame({
            'Mã': [ticker], 
            'Giá': [f"{last_row['Close']:,.0f}"], 
            'RSI': [f"{last_row['RSI']:.1f}"]
        })])
    st.table(report_df.set_index('Mã'))

with col_rep2:
    st.success("Nhận định AI")
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
        t_df = calculate_technical_indicators(t_df)
        with st.expander(f"Chi tiết {ticker}", expanded=True):
            st.markdown(generate_insight(ticker, t_df))
