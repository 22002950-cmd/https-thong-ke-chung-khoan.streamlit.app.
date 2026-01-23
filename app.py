import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Vietnam Market Deep Dive",
    layout="wide",
    page_icon="",
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

# --- 2. HÀM XỬ LÝ DỮ LIỆU NÂNG CAO ---

def clean_currency(val):
    """Làm sạch chuỗi tiền tệ (vd: '23,450.50 ₫' -> 23450.5)"""
    if pd.isna(val) or val == '':
        return np.nan
    # Chuyển về string, bỏ '₫', bỏ dấu phẩy, strip khoảng trắng
    clean_str = str(val).replace('₫', '').replace(',', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return np.nan

@st.cache_data
def load_and_merge_data():
    """Đọc 4 file CSV, làm sạch và gộp thành chuẩn Long Format"""
    
    # Định nghĩa map file và ngành
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
            # Đọc file raw
            df_raw = pd.read_csv(filename)
            
            # Chuẩn hóa tên cột: bỏ khoảng trắng thừa, viết thường
            df_raw.columns = [c.strip() for c in df_raw.columns]
            
            # Tìm cột thời gian (thường là 'time')
            time_col = next((c for c in df_raw.columns if 'time' in c.lower()), None)
            if not time_col:
                continue
                
            # Xử lý ngày tháng (Quan trọng: ép kiểu MDY nếu cần)
            df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors='coerce')
            
            # Lấy danh sách các Mã (Ticker) từ tên cột
            # Logic: Cột thường có dạng "TICKER_price_close" hoặc "TICKER_Price_Close"
            # Ta sẽ lọc ra các cột chứa "close"
            close_cols = [c for c in df_raw.columns if 'close' in c.lower()]
            
            for col in close_cols:
                # Tách tên mã từ tên cột (vd: "SSI_price_close" -> "SSI")
                # Dùng Regex để lấy phần đầu trước dấu _ hoặc khoảng trắng
                ticker_match = re.split(r'[_ ]', col)[0].upper()
                
                # Bỏ qua nếu không phải ticker hợp lệ (vd "Price", "Mã")
                if len(ticker_match) > 4 or len(ticker_match) < 3: 
                    continue
                    
                # Tạo dataframe con cho từng mã
                temp_df = pd.DataFrame()
                temp_df['Date'] = df_raw[time_col]
                temp_df['Close'] = df_raw[col].apply(clean_currency)
                temp_df['Ticker'] = ticker_match
                temp_df['Sector'] = sector_name
                
                # Tìm cột Volume tương ứng
                # Thường là TICKER_volume
                vol_col = next((c for c in df_raw.columns if ticker_match in c.upper() and 'volume' in c.lower()), None)
                if vol_col:
                    temp_df['Volume'] = df_raw[vol_col].apply(clean_currency) # Volume cũng có thể dính dấu phẩy
                else:
                    temp_df['Volume'] = 0
                
                all_data.append(temp_df)
                
        except Exception as e:
            st.error(f"Lỗi khi xử lý file {filename}: {str(e)}")

    if not all_data:
        return pd.DataFrame()
        
    # Gộp tất cả lại
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.dropna(subset=['Date', 'Close'])
    df_final = df_final.sort_values(['Ticker', 'Date'])
    
    return df_final

# --- 3. HÀM TÍNH TOÁN CHỈ SỐ KỸ THUẬT ---

def calculate_technical_indicators(df):
    """Tính RSI, MA, Bollinger Bands"""
    df = df.copy()
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (std * 2)
    df['Lower_Band'] = df['MA20'] - (std * 2)
    
    # Drawdown (Sụt giảm từ đỉnh)
    df['Peak'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
    
    return df

# --- 4. GIAO DIỆN CHÍNH ---

df = load_and_merge_data()

if df.empty:
    st.error(" Không tìm thấy dữ liệu hoặc file không đúng định dạng. Hãy upload 4 file CSV vào cùng thư mục.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("🛠️ BỘ LỌC DỮ LIỆU")

# Chọn ngày
min_date, max_date = df['Date'].min(), df['Date'].max()
start_date = st.sidebar.date_input("Từ ngày", min_date)
end_date = st.sidebar.date_input("Đến ngày", max_date)

# Chọn ngành
sectors = df['Sector'].unique()
selected_sectors = st.sidebar.multiselect("Chọn Ngành", sectors, default=sectors)

# Lọc sơ bộ theo ngành
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                 (df['Date'] <= pd.to_datetime(end_date)) & 
                 (df['Sector'].isin(selected_sectors))]

# Chọn Cổ phiếu (dựa trên ngành đã chọn)
available_tickers = df_filtered['Ticker'].unique()
selected_tickers = st.sidebar.multiselect("Chọn Mã (Max 5)", available_tickers, default=available_tickers[:3])

if not selected_tickers:
    st.warning("Vui lòng chọn ít nhất 1 mã cổ phiếu.")
    st.stop()

df_display = df_filtered[df_filtered['Ticker'].isin(selected_tickers)]

# --- MAIN CONTENT ---
st.title("TRADING ANALYTICS")
st.markdown("Phân tích dữ liệu đa ngành: Chứng khoán, Thép, Bank, Bất động sản")
st.markdown("---")

# 1. KPI CARDS
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

# 2. TAB PHÂN TÍCH
tab1, tab2, tab3, tab4 = st.tabs(["📈 Sức Mạnh Giá (Trend)", "🔬 Phân Tích Kỹ Thuật", "❄️ Mùa Vụ (Heatmap)", "⚠️ Rủi Ro & Drawdown"])

# --- TAB 1: SO SÁNH SỨC MẠNH ---
with tab1:
    st.markdown("##### So sánh tăng trưởng (%) từ đầu kỳ")
    
    # Pivot để vẽ biểu đồ line
    pivot = df_display.pivot(index='Date', columns='Ticker', values='Close')
    # Chuẩn hóa về %
    pivot_norm = pivot.apply(lambda x: (x / x.iloc[0] - 1) * 100)
    
    fig_norm = px.line(pivot_norm, x=pivot_norm.index, y=pivot_norm.columns,
                      labels={'value': 'Tăng trưởng (%)', 'variable': 'Mã'},
                      template='plotly_dark')
    fig_norm.update_layout(hovermode="x unified", height=500, plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_norm, use_container_width=True)

# --- TAB 2: PHÂN TÍCH KỸ THUẬT CHI TIẾT ---
with tab2:
    target_ticker = st.selectbox("Chọn mã để soi kỹ thuật:", selected_tickers)
    
    # Lấy dữ liệu và tính toán chỉ báo
    tech_df = df[df['Ticker'] == target_ticker].copy()
    tech_df = tech_df.sort_values('Date')
    tech_df = calculate_technical_indicators(tech_df)
    
    # Lọc lại theo ngày user chọn
    mask = (tech_df['Date'] >= pd.to_datetime(start_date)) & (tech_df['Date'] <= pd.to_datetime(end_date))
    tech_view = tech_df.loc[mask]
    
    # Chart 1: Giá + BB + MA
    fig_tech = go.Figure()
    # Giá
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Close'], name='Close', line=dict(color='white', width=1)))
    # Bollinger Bands
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Upper_Band'], name='Upper BB', line=dict(color='gray', width=0), showlegend=False))
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['Lower_Band'], name='Lower BB', line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', showlegend=False))
    # MA
    fig_tech.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['MA20'], name='MA20', line=dict(color='yellow', width=1)))
    
    fig_tech.update_layout(title=f"Phân tích giá {target_ticker}", template='plotly_dark', height=400, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_tech, use_container_width=True)
    
    # Chart 2: RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=tech_view['Date'], y=tech_view['RSI'], name='RSI', line=dict(color='#a78bfa')))
    # Đường 30 và 70
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Quá mua")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Quá bán")
    fig_rsi.update_layout(title="Chỉ số RSI (Sức mạnh tương đối)", template='plotly_dark', height=250, yaxis=dict(range=[0, 100]), margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_rsi, use_container_width=True)

# --- TAB 3: MÙA VỤ (HEATMAP) ---
with tab3:
    st.markdown("##### Hiệu suất trung bình theo Tháng (Seasonality)")
    st.info("Biểu đồ này giúp bạn trả lời: Mã này thường tăng mạnh vào tháng mấy?")
    
    # Tính % thay đổi theo tháng cho TOÀN BỘ lịch sử (để thống kê chính xác)
    full_history = df[df['Ticker'].isin(selected_tickers)].copy()
    full_history['Month'] = full_history['Date'].dt.month
    full_history['Year'] = full_history['Date'].dt.year
    
    # Pivot Monthly Return
    # Bước 1: Tính giá cuối tháng
    monthly_close = full_history.groupby(['Ticker', 'Year', 'Month'])['Close'].last().reset_index()
    monthly_close['Pct_Change'] = monthly_close.groupby('Ticker')['Close'].pct_change() * 100
    
    # Bước 2: Tính trung bình theo tháng
    seasonality = monthly_close.groupby(['Ticker', 'Month'])['Pct_Change'].mean().reset_index()
    
    # Heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=seasonality['Pct_Change'],
        x=seasonality['Month'],
        y=seasonality['Ticker'],
        colorscale='RdBu', # Đỏ giảm, Xanh tăng
        zmid=0,
        text=seasonality['Pct_Change'],
        texttemplate="%{text:.1f}%"
    ))
    
    fig_heat.update_layout(
        template='plotly_dark',
        height=400,
        xaxis=dict(title='Tháng', tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title='Mã')
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: RỦI RO (DRAWDOWN) ---
with tab4:
    st.markdown("##### Drawdown: Mức độ sụt giảm từ đỉnh gần nhất")
    st.caption("Chỉ số này cho biết nếu bạn lỡ 'đu đỉnh' thì tài khoản đang âm bao nhiêu %.")
    
    # Tính Drawdown cho các mã đã chọn (trong khoảng thời gian lọc)
    dd_data = pd.DataFrame()
    
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].sort_values('Date').copy()
        # Tính Drawdown trên toàn lịch sử trước
        t_df['Peak'] = t_df['Close'].cummax()
        t_df['Drawdown'] = (t_df['Close'] - t_df['Peak']) / t_df['Peak'] * 100
        # Lọc lại theo ngày hiển thị
        mask_dd = (t_df['Date'] >= pd.to_datetime(start_date)) & (t_df['Date'] <= pd.to_datetime(end_date))
        dd_data[ticker] = t_df.loc[mask_dd].set_index('Date')['Drawdown']
    
    # Vẽ Area Chart
    fig_dd = px.area(dd_data, x=dd_data.index, y=dd_data.columns,
                     labels={'value': 'Drawdown (%)', 'variable': 'Mã'},
                     template='plotly_dark')
    fig_dd.update_layout(height=400, yaxis=dict(title='Sụt giảm (%)'))
    st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("---")

st.caption("Data source: Combined Sector Files | Processed by Python & Streamlit")
            # --- 5. TÍNH NĂNG: XUẤT BÁO CÁO TỰ ĐỘNG (AUTO REPORT) ---
st.markdown("---")
st.header(" Báo Cáo Phân Tích Tự Động")
st.caption("Hệ thống tự động quét dữ liệu và đưa ra khuyến nghị dựa trên chỉ báo kỹ thuật.")

# --- 5. TÍNH NĂNG: XUẤT BÁO CÁO TỰ ĐỘNG (AUTO REPORT) ---
st.markdown("---")
st.header(" Báo Cáo Phân Tích Tự Động")
st.caption("Hệ thống tự động quét dữ liệu và đưa ra khuyến nghị dựa trên chỉ báo kỹ thuật.")

# Hàm tạo câu nhận xét
def generate_insight(ticker, df_input):
    # Lấy dữ liệu mới nhất
    last_row = df_input.iloc[-1]
    prev_row = df_input.iloc[-2]
    
    # 1. Xu hướng (Trend)
    trend = "TĂNG" if last_row['Close'] > last_row['MA20'] else "GIẢM"
    trend_icon = "🟢" if trend == "TĂNG" else "🔴"
    
    # 2. Động lượng (RSI)
    rsi = last_row['RSI']
    rsi_signal = "Trung tính"
    if rsi > 70: rsi_signal = "QUÁ MUA (Nguy hiểm )"
    elif rsi < 30: rsi_signal = "QUÁ BÁN (Cơ hội bắt đáy )"
    
    # 3. Biến động (Bollinger Bands)
    bb_signal = "Bình thường"
    if last_row['Close'] > last_row['Upper_Band']:
        bb_signal = "Vượt dải trên (Đột biến giá)"
    elif last_row['Close'] < last_row['Lower_Band']:
        bb_signal = "Thủng dải dưới (Rơi mạnh)"
        
    return f"""
    **Mã: {ticker}** ({last_row['Date'].strftime('%d/%m/%Y')})
    - **Giá đóng cửa:** {last_row['Close']:,.0f} VND
    - **Xu hướng ngắn hạn:** {trend_icon} Đang trong xu hướng {trend} (Giá {'trên' if trend=='TĂNG' else 'dưới'} MA20).
    - **Trạng thái RSI:** {rsi:.1f} - {rsi_signal}.
    - **Tín hiệu Bollinger:** {bb_signal}.
    - **Khuyến nghị Robot:** {'Canh chốt lời dần' if rsi > 70 else ('Xem xét giải ngân' if rsi < 30 else 'Nắm giữ / Quan sát thêm')}.
    """

# Hiển thị báo cáo
col1, col2 = st.columns([1, 1])

with col1:
    st.info( **Tóm tắt Chỉ số Kỹ thuật**)
    report_df = pd.DataFrame()
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
        t_df = calculate_technical_indicators(t_df)
        last_row = t_df.iloc[-1]
        
        report_df = pd.concat([report_df, pd.DataFrame({
            'Mã': [ticker],
            'Giá': [f"{last_row['Close']:,.0f}"],
            'RSI': [f"{last_row['RSI']:.1f}"],
            'MA20': [f"{last_row['MA20']:,.0f}"]
        })])
    st.table(report_df.set_index('Mã'))

with col2:
    st.success( **Nhận định chi tiết (AI Rule-based)**)
    for ticker in selected_tickers:
        t_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
        t_df = calculate_technical_indicators(t_df)
        
        with st.expander(f"Xem chi tiết mã {ticker}", expanded=True):
            st.markdown(generate_insight(ticker, t_df))
