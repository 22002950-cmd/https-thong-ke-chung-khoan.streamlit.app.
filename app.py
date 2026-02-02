import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Vietnam Market Analytics",
    layout="wide",
    page_icon="📈",
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

# --- HÀM CẬP NHẬT GIÁ (AUTO UPDATE) ---
def update_realtime_data(df_historical):
    if df_historical.empty:
        return df_historical

    last_date = df_historical['Date'].max()
    today = datetime.now()
    
    if last_date.date() >= (today - timedelta(days=1)).date():
        return df_historical

    st.toast(f"🔄 Đang cập nhật dữ liệu từ {last_date.date()} đến nay...", icon="☁️")

    ticker_sector_map = df_historical.set_index('Ticker')['Sector'].to_dict()
    tickers = list(ticker_sector_map.keys())
    yf_tickers = [t + ".VN" for t in tickers]
    
    try:
        start_fetch = last_date + timedelta(days=1)
        if start_fetch.date() > today.date():
             return df_historical

        new_data = yf.download(yf_tickers, start=start_fetch, progress=False)
        
        if new_data.empty:
            return df_historical
            
        if 'Close' in new_data.columns:
            df_close = new_data['Close'].reset_index()
            if len(tickers) == 1:
                df_close.columns = ['Date', tickers[0]] 
            
            df_melted = df_close.melt(id_vars=['Date'], var_name='Ticker', value_name='Close')
        else:
            return df_historical
            
        if 'Volume' in new_data.columns:
            df_vol = new_data['Volume'].reset_index()
            if len(tickers) == 1:
                df_vol.columns = ['Date', tickers[0]]
                
            df_vol_melted = df_vol.melt(id_vars=['Date'], var_name='Ticker', value_name='Volume')
            df_melted = pd.merge(df_melted, df_vol_melted, on=['Date', 'Ticker'], how='left')
        else:
            df_melted['Volume'] = 0

        df_melted['Ticker'] = df_melted['Ticker'].str.replace('.VN', '', regex=False)
        df_melted['Sector'] = df_melted['Ticker'].map(ticker_sector_map)
        df_melted['Date'] = pd.to_datetime(df_melted['Date'])
        df_melted = df_melted.dropna(subset=['Close'])

        df_final = pd.concat([df_historical, df_melted], ignore_index=True)
        df_final = df_final.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        st.toast("✅ Đã cập nhật xong dữ liệu mới nhất!", icon="✅")
        return df_final

    except Exception as e:
        st.toast(f"Lỗi cập nhật: {str(e)}", icon="⚠️")
        return df_historical

# --- 3. TÍNH TOÁN CHỈ SỐ KỸ THUẬT & MÔ PHỎNG ---

def calculate_technical_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    
    df['Peak'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
    
    return df

def monte_carlo_simulation(df, ticker, forecast_days=30, simulations=500):
    """Mô phỏng Monte Carlo dự báo giá"""
    df_ticker = df[df['Ticker'] == ticker].sort_values('Date')
    if len(df_ticker) < 30:
        return None, None
        
    last_price = df_ticker['Close'].iloc[-1]
    
    # Tính log returns để có độ biến động chuẩn hơn
    log_returns = np.log(1 + df_ticker['Close'].pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    
    # Drift (Xu hướng) và Shock (Biến động ngẫu nhiên)
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    # Tạo ma trận dự báo
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (forecast_days, simulations)))
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    
    for t in range(1, forecast_days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths, df_ticker['Date'].iloc[-1]

# --- 4. GIAO DIỆN CHÍNH ---

df = load_and_merge_data()

with st.spinner('Đang kiểm tra và cập nhật dữ liệu thị trường mới nhất...'):
    df = update_realtime_data(df)

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
default_tickers = available_tickers[:3] if len(available_tickers) > 0 else []
selected_tickers = st.sidebar.multiselect("Chọn Mã (Max 5)", available_tickers, default=default_tickers)

if not selected_tickers:
    st.warning("Vui lòng chọn ít nhất 1 mã cổ phiếu.")
    st.stop()

df_display = df_filtered[df_filtered['Ticker'].isin(selected_tickers)]

# --- MAIN PAGE ---
st.title("PRO TRADING ANALYTICS")
st.markdown("---")

cols = st.columns(len(selected_tickers))
for i, ticker in enumerate(selected_tickers):
    ticker_data = df_display[df_display['Ticker'] == ticker]
    if not ticker_data.empty:
        curr_price = ticker_data.iloc[-1]['Close']
        prev_price = ticker_data.iloc[-2]['Close'] if len(ticker_data) > 1 else curr_price
        change = ((curr_price - prev_price) / prev_price) * 100
        
        color_cls = "gain" if change >= 0 else "loss"
        symbol = "▲" if change >= 0 else "▼"
        
        if i < len(cols):
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
# Thêm Tab 5: Dự Phóng Tương Lai
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sức Mạnh Giá", "Phân Tích Kỹ Thuật", "Xu Hướng Mùa Vụ", "Rủi Ro Drawdown", "🔮 Dự Phóng Tương Lai"])

with tab1:
    st.markdown("##### So sánh tăng trưởng (%) từ đầu kỳ")
    if not df_display.empty:
        pivot = df_display.pivot(index='Date', columns='Ticker', values='Close')
        pivot_norm = pivot.apply(lambda x: (x / x.iloc[0] - 1) * 100 if x.iloc[0] != 0 else 0)
        fig_norm = px.line(pivot_norm, x=pivot_norm.index, y=pivot_norm.columns, template='plotly_dark')
        st.plotly_chart(fig_norm, use_container_width=True)
    else:
        st.info("Chưa đủ dữ liệu để vẽ biểu đồ.")

with tab2:
    target_ticker = st.selectbox("Chọn mã soi chi tiết (KT):", selected_tickers, key="tech_select")
    tech_df = df[df['Ticker'] == target_ticker].copy().sort_values('Date')
    
    if len(tech_df) > 0:
        tech_df = calculate_technical_indicators(tech_df)
        mask = (tech_df['Date'] >= pd.to_datetime(start_date)) & (tech_df['Date'] <= pd.to_datetime(end_date))
        tech_view = tech_df.loc[mask]
        
        fig_tech = go.Figure()
        fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Close'], name='Close', line=dict(color='white')))
        fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['MA20'], name='MA20', line=dict(color='yellow')))
        fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Upper_Band'], line=dict(color='gray', width=0), showlegend=False))
        fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Lower_Band'], line=dict(color='gray', width=0), fill='tonexty', showlegend=False))
        fig_tech.update_layout(template='plotly_dark', title=f"Biểu đồ giá {target_ticker}")
        st.plotly_chart(fig_tech, use_container_width=True)
        
        fig_rsi = px.line(tech_view, x='Date', y='RSI', title="Chỉ số RSI")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(template='plotly_dark', height=250)
        st.plotly_chart(fig_rsi, use_container_width=True)

with tab3:
    st.markdown("##### Xu hướng Mùa vụ (Theo tháng)")
    full_history = df[df['Ticker'].isin(selected_tickers)].copy()
    if not full_history.empty:
        full_history['Month'] = full_history['Date'].dt.month
        full_history['Year'] = full_history['Date'].dt.year
        monthly_close = full_history.groupby(['Ticker', 'Year', 'Month'])['Close'].last().reset_index()
        monthly_close['Pct_Change'] = monthly_close.groupby('Ticker')['Close'].pct_change()
        seasonality_avg = monthly_close.groupby(['Ticker', 'Month'])['Pct_Change'].mean().reset_index()
        
        season_chart_data = []
        for ticker in selected_tickers:
            t_data = seasonality_avg[seasonality_avg['Ticker'] == ticker].sort_values('Month')
            t_data['Cumulative_Trend'] = (1 + t_data['Pct_Change'].fillna(0)).cumprod() * 100
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
        if not t_df.empty:
            t_df['Peak'] = t_df['Close'].cummax()
            t_df['Drawdown'] = (t_df['Close'] - t_df['Peak']) / t_df['Peak'] * 100
            mask_dd = (t_df['Date'] >= pd.to_datetime(start_date)) & (t_df['Date'] <= pd.to_datetime(end_date))
            t_subset = t_df.loc[mask_dd]
            if not t_subset.empty:
                dd_data[ticker] = t_subset.set_index('Date')['Drawdown']
    
    if not dd_data.empty:
        fig_dd = px.area(dd_data, template='plotly_dark')
        st.plotly_chart(fig_dd, use_container_width=True)

# --- TAB 5: MONTE CARLO SIMULATION ---
with tab5:
    st.markdown("##### Mô phỏng Monte Carlo: Dự báo 1000 kịch bản giá tương lai")
    st.info("ℹ️ Đây là mô phỏng xác suất dựa trên biến động lịch sử, không phải lời khuyên đầu tư chắc chắn.")
    
    mc_col1, mc_col2 = st.columns([1, 3])
    
    with mc_col1:
        mc_ticker = st.selectbox("Chọn mã dự báo:", selected_tickers)
        forecast_weeks = st.slider("Số tuần dự báo:", min_value=1, max_value=24, value=4)
        forecast_days = forecast_weeks * 5 # Quy đổi ra ngày giao dịch
        
    price_paths, last_date = monte_carlo_simulation(df, mc_ticker, forecast_days)
    
    if price_paths is not None:
        # Xử lý dữ liệu vẽ biểu đồ
        future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
        
        # Lấy giá trị trung vị (Median) và các khoảng tin cậy
        median_path = np.median(price_paths, axis=1)
        upper_bound = np.percentile(price_paths, 95, axis=1) # Kịch bản lạc quan (Top 5%)
        lower_bound = np.percentile(price_paths, 5, axis=1)  # Kịch bản bi quan (Bottom 5%)
        
        # Vẽ biểu đồ
        fig_mc = go.Figure()
        
        # 1. Vẽ dữ liệu lịch sử gần đây (60 ngày cuối)
        hist_df = df[df['Ticker'] == mc_ticker].sort_values('Date').tail(60)
        fig_mc.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Close'], mode='lines', 
                                    name='Lịch sử (60 phiên)', line=dict(color='white', width=2)))
        
        # 2. Vẽ vùng dự báo (Dải tin cậy)
        fig_mc.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode='lines', 
                                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_mc.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode='lines', 
                                    line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', 
                                    name='Vùng biến động (90% xác suất)'))
        
        # 3. Vẽ đường trung vị
        fig_mc.add_trace(go.Scatter(x=future_dates, y=median_path, mode='lines', 
                                    name='Dự báo Trung vị', line=dict(color='yellow', dash='dash')))
        
        fig_mc.update_layout(template='plotly_dark', title=f"Dự báo {mc_ticker} trong {forecast_weeks} tuần tới")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Hiển thị bảng số liệu chốt
        last_pred_median = median_path[-1]
        last_pred_upper = upper_bound[-1]
        last_pred_lower = lower_bound[-1]
        curr_price_mc = hist_df.iloc[-1]['Close']
        
        st.markdown(f"""
        **Kết quả dự báo sau {forecast_weeks} tuần:**
        - Giá hiện tại: **{curr_price_mc:,.0f}**
        - Kịch bản Trung bình (Median): **{last_pred_median:,.0f}** ({((last_pred_median-curr_price_mc)/curr_price_mc)*100:+.2f}%)
        - Kịch bản Lạc quan (Top 5%): <span style='color:#4ade80'>**{last_pred_upper:,.0f}**</span>
        - Kịch bản Bi quan (Bottom 5%): <span style='color:#f87171'>**{last_pred_lower:,.0f}**</span>
        """, unsafe_allow_html=True)

# --- 5. BÁO CÁO TỰ ĐỘNG ---
st.markdown("---")
st.header("Báo Cáo Phân Tích Tự Động")

def generate_insight(ticker, df_input):
    if df_input.empty: return "Chưa đủ dữ liệu"
    last_row = df_input.iloc[-1]
    trend = "TĂNG" if last_row['Close'] > last_row['MA20'] else "GIẢM"
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
        if not t_df.empty:
            t_df = calculate_technical_indicators(t_df)
            last_row = t_df.iloc[-1]
            report_df = pd.concat([report_df, pd.DataFrame({
                'Mã': [ticker], 
                'Giá': [f"{last_row['Close']:,.0f}"], 
                'RSI': [f"{last_row['RSI']:.1f}"]
            })])
    if not report_df.empty:
        st.table(report_df.set_index('Mã'))

with col_rep2:
    st.success("Nhận định AI")
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
        if not t_df.empty:
            t_df = calculate_technical_indicators(t_df)
            with st.expander(f"Chi tiết {ticker}", expanded=True):
                st.markdown(generate_insight(ticker, t_df))
