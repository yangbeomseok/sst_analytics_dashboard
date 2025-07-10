import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import re
import matplotlib.pyplot as plt

# --- 데이터 로드 및 모델 학습 (캐시 사용) ---
@st.cache_data
def load_model_and_data():
    print("✅ 데이터 로드 및 모델 학습을 시작합니다... (이 메시지는 처음 한 번만 보여야 합니다)")
    train_df = pd.read_csv('train_data.csv', index_col='valid_time', parse_dates=True)
    validation_df = pd.read_csv('validation_data.csv', index_col='valid_time', parse_dates=True)
    test_df = pd.read_csv('test_data.csv', index_col='valid_time', parse_dates=True)

    def make_lag_features(df, lag_days):
        df_lag = df.copy()
        cols_to_lag = [col for col in df_lag.columns if 'sst' in col or 't2m' in col or 'wind' in col]
        for col in cols_to_lag:
            for lag in lag_days:
                df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag * 4)
        return df_lag

    lag_days = [1, 2, 3, 7]
    full_train_df = pd.concat([train_df, validation_df])
    train_featured = make_lag_features(full_train_df, lag_days)
    test_featured = make_lag_features(test_df, lag_days)
    train_featured = train_featured.dropna()
    test_featured = test_featured.dropna()

    def split_X_y(df):
        X = df.drop(columns=['sst', 'number', 'latitude', 'longitude', 'expver'])
        y = df['sst']
        return X, y

    X_train, y_train = split_X_y(train_featured)
    X_test, y_test = split_X_y(test_featured)
    
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    test_predictions = pd.Series(model.predict(X_test), index=X_test.index)
    
    return model, X_test, y_test, test_predictions

# --- 앱 실행 시 모델과 데이터 로드 ---
model, X_test, y_test, test_predictions = load_model_and_data()

# --- Streamlit 웹페이지 UI 구성 ---
st.set_page_config(layout="wide")
st.title('🌊 AI 해수면 온도(SST) 예측기')

st.info("본 모델은 2024년의 데이터에 대한 예측을 수행합니다.")

# --- 사이드바 (입력 부분) ---
st.sidebar.header("🗓️ 날짜 선택")

if 'date_input' not in st.session_state:
    st.session_state.date_input = "2024년 8월 15일 14시"

def set_summer_example():
    st.session_state.date_input = "2024년 8월 15일 14시"

def set_winter_example():
    st.session_state.date_input = "2024년 1월 20일 10시"

st.sidebar.text_input("날짜와 시간을 입력하세요", key="date_input")
st.sidebar.write("클릭으로 예시 날짜를 입력할 수 있습니다.")
col1, col2 = st.sidebar.columns(2)
col1.button("여름 예시 (8월)", on_click=set_summer_example)
col2.button("겨울 예시 (1월)", on_click=set_winter_example)

predict_button = st.sidebar.button('예측 실행', type="primary")

# --- 메인 페이지 (결과 부분) ---
st.subheader("모델 예측 결과")

if predict_button:
    date_str = st.session_state.date_input
    try:
        numbers = re.findall(r'\d+', date_str)
        if len(numbers) < 4:
            raise ValueError("날짜/시간 정보가 부족합니다.")
        
        year, month, day, hour = numbers[:4]
        minute = numbers[4] if len(numbers) > 4 else '00'
        standard_format_str = f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}"
        target_time = pd.to_datetime(standard_format_str)

        min_date, max_date = y_test.index.min(), y_test.index.max()
        if not (min_date <= target_time <= max_date):
            st.error(f"예측 가능 범위를 벗어났습니다. (기간: {min_date.date()} ~ {max_date.date()})")
        else:
            time_diff = (y_test.index - target_time).to_series().abs()
            closest_index_pos = np.argmin(time_diff.values)
            closest_time = y_test.index[closest_index_pos]

            input_features = X_test.iloc[[closest_index_pos]]
            actual_temp = y_test.iloc[closest_index_pos]
            
            predicted_temp = model.predict(input_features)[0]
            error = actual_temp - predicted_temp

            st.success(f"**{closest_time.strftime('%Y년 %m월 %d일 %H시')}**의 예측 결과입니다.")

            # ★★★ 입력 시간과 실제 예측 시간이 다를 경우 안내 메시지 추가 ★★★
            if target_time != closest_time:
                st.info(f"입력하신 시간 '{target_time.strftime('%H:%M')}'의 데이터가 없어, 가장 가까운 시간인 '{closest_time.strftime('%H:%M')}'의 결과가 표시됩니다.")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ 모델 예측 온도", f"{predicted_temp:.2f} °C")
            col2.metric("🎯 실제 정답 온도", f"{actual_temp:.2f} °C")
            col3.metric("📊 오차", f"{error:.2f} °C", delta_color="inverse")

            st.write("---")
            st.subheader(f"'{closest_time.date()}' 주변 예측 추세 그래프")
            
            start_date = closest_time - pd.Timedelta(days=3)
            end_date = closest_time + pd.Timedelta(days=3)

            chart_data_actual = y_test.loc[start_date:end_date]
            chart_data_to_predict = X_test.loc[start_date:end_date]
            chart_predictions = model.predict(chart_data_to_predict)

            chart_df = pd.DataFrame({
                'Actual SST (실제값)': chart_data_actual,
                'Predicted SST (예측값)': chart_predictions
            }, index=chart_data_actual.index)

            st.line_chart(chart_df)

    except Exception as e:
        st.error(f"입력 형식이 잘못되었습니다. 다시 확인해주세요. (에러: {e})")

# --- 전체 성능 분석 대시보드 ---
st.write("---")
with st.expander("Click ! 📈 전체 모델 성능 분석 대시보드 보기"):
    st.markdown("<p style='font-size: 14px;'>아래 그래프들은 2024년 전체 테스트 데이터에 대한 모델의 종합 성능을 보여줍니다.</p>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["시계열 추세", "오차 분포도", "예측-실제 산점도", "월별 오차", "특성 중요도"])

    with tab1:
        chart_df_full = pd.DataFrame({'Actual SST (실제값)': y_test, 'Predicted SST (예측값)': test_predictions})
        st.line_chart(chart_df_full)

    with tab2:
        errors = y_test - test_predictions
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title('Prediction Error Distribution', fontsize=10)
        ax.set_xlabel('Prediction Error (°C)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, test_predictions, alpha=0.5, s=5)
        ax.set_title('Actual vs. Predicted SST', fontsize=10)
        ax.set_xlabel('Actual Temperature (°C)', fontsize=8)
        ax.set_ylabel('Predicted Temperature (°C)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
        ax.grid(True)
        st.pyplot(fig)

    with tab4:
        monthly_errors = pd.DataFrame({'month': y_test.index.month, 'error': y_test - test_predictions})
        fig, ax = plt.subplots(figsize=(8, 5))
        monthly_errors.boxplot(by='month', column='error', ax=ax, grid=False)
        ax.set_title('Monthly Prediction Error', fontsize=10)
        ax.set_xlabel('Month', fontsize=8)
        ax.set_ylabel('Prediction Error (°C)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.suptitle('')
        st.pyplot(fig)

    with tab5:
        fig, ax = plt.subplots(figsize=(8, 6))
        lgb.plot_importance(model, ax=ax, max_num_features=15)
        ax.set_title('Feature Importance', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        st.pyplot(fig)
