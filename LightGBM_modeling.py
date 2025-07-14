import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
# analysis.py에서 전처리한 CSV 파일 로드
train_df = pd.read_csv('train_data.csv', index_col='valid_time', parse_dates=True)
validation_df = pd.read_csv('validation_data.csv', index_col='valid_time', parse_dates=True)
test_df = pd.read_csv('test_data.csv', index_col='valid_time', parse_dates=True)


# Feature Engineering: 시차(Lag) 특성 생성
def make_lag_features(df, lag_days):
    """과거 데이터를 새로운 특성으로 추가"""
    df_lag = df.copy()
    # sst, t2m, wind 관련 변수만 선택
    cols_to_lag = [col for col in df_lag.columns if 'sst' in col or 't2m' in col or 'wind' in col]
    for col in cols_to_lag:
        for lag in lag_days:
            # lag * 4: 데이터가 6시간 간격이므로 1일은 4 step
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag * 4)
    return df_lag

# 시차를 적용할 날짜 설정: 1, 2, 3, 7일 전
lag_days = [1, 2, 3, 7]
train_featured = make_lag_features(train_df, lag_days)
validation_featured = make_lag_features(validation_df, lag_days)
test_featured = make_lag_features(test_df, lag_days)

# 시차 특성 생성으로 인해 발생한 결측치(NaN) 제거
train_featured = train_featured.dropna()
validation_featured = validation_featured.dropna()
test_featured = test_featured.dropna()


# 데이터 분리 (X, y)
def split_X_y(df):
    """데이터를 입력(X)과 정답(y)으로 분리"""
    # 타겟('sst')과 불필요한 식별자 컬럼 제외
    X = df.drop(columns=['sst', 'number', 'latitude', 'longitude', 'expver'])
    # 예측 목표 'sst'
    y = df['sst']
    return X, y

X_train, y_train = split_X_y(train_featured)
X_validation, y_validation = split_X_y(validation_featured)
X_test, y_test = split_X_y(test_featured)


# 모델 학습 (LightGBM)
# PPT: "Model Training" 파트 핵심 로직
# LightGBM 모델 초기화 (재현성을 위해 random_state 고정)
model = lgb.LGBMRegressor(random_state=42)

print("모델 학습 시작...")
# 모델 훈련
model.fit(X_train, y_train)
print("모델 학습 완료")


# 모델 성능 평가 및 결과 시각화
print("\n[테스트 데이터 성능]")
test_predictions = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f"MAE: {test_mae:.4f} °C")
print(f"RMSE: {test_rmse:.4f} °C")

# 1. 실제값 vs 예측값 시계열 그래프
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(y_test.index, y_test, label='Actual SST (실제값)')
ax.plot(y_test.index, test_predictions, label='Predicted SST (예측값)', linestyle='--', alpha=0.7)
ax.set_title('Final SST Prediction vs Actual (Test Set)', fontsize=16)
ax.legend()
plt.show()

# 2. 특성 중요도
lgb.plot_importance(model, figsize=(10, 8), max_num_features=15)
plt.title('Feature Importance', fontsize=16)
plt.show()

# 3. 오차 분포 (히스토그램)
errors = y_test - test_predictions
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.title('Error Distribution (Test Set)', fontsize=16)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2) # 오차 0인 지점 표시
plt.show()

# 4. 실제값 vs 예측값 산점도
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.5, s=10)
plt.title('Actual vs. Predicted SST (Test Set)', fontsize=16)
# y=x 참조선 (완벽한 예측일 경우 모든 점이 이 선 위에 위치)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.grid(True)
plt.show()

# 5. 월별 오차 (박스플롯)
monthly_errors = pd.DataFrame({'month': y_test.index.month, 'error': errors})
plt.figure(figsize=(12, 7))
monthly_errors.boxplot(by='month', column='error', grid=False)
plt.title('Monthly Prediction Error Distribution (Test Set)', fontsize=16)
plt.suptitle('') # 자동 생성되는 상단 타이틀 제거
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.show()
