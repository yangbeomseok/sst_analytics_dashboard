import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# --- 데이터 로드 ---
# analysis.py에서 전처리 후 저장한 CSV 파일들을 불러옵니다.
train_df = pd.read_csv('train_data.csv', index_col='valid_time', parse_dates=True)
validation_df = pd.read_csv('validation_data.csv', index_col='valid_time', parse_dates=True)
test_df = pd.read_csv('test_data.csv', index_col='valid_time', parse_dates=True)


# --- Feature Engineering ---
def make_lag_features(df, lag_days):
    """
    "Feature Engineering" 파트 핵심 로직
    모델이 과거의 패턴을 학습할 수 있도록, 과거 N일 전의 주요 변수 값을
    '힌트'가 되는 새로운 특성으로 생성한다.
    """
    df_lag = df.copy()
    # 주요 변수(수온, 기온, 풍속)들에 대해 시차 특성을 생성합니다.
    cols_to_lag = [col for col in df_lag.columns if 'sst' in col or 't2m' in col or 'wind' in col]
    for col in cols_to_lag:
        for lag in lag_days:
            # 예: 1일 전(lag=1) 데이터는 4 step 전의 값을 가져옴 (6시간 간격 데이터)
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag * 4)
    return df_lag

# 1일, 2일, 3일, 7일 전의 과거 데이터를 '힌트'로 사용하도록 설정합니다.
lag_days = [1, 2, 3, 7]
train_featured = make_lag_features(train_df, lag_days)
validation_featured = make_lag_features(validation_df, lag_days)
test_featured = make_lag_features(test_df, lag_days)

# 시차 특성 생성 시, 맨 앞부분의 데이터에는 과거 정보가 없어 결측치(NaN)가 발생하므로 제거합니다.
train_featured = train_featured.dropna()
validation_featured = validation_featured.dropna()
test_featured = test_featured.dropna()


# --- 데이터 분리 (입력 데이터와 정답 데이터) ---
def split_X_y(df):
    """데이터를 모델의 입력(X)과 우리가 예측할 정답(y)으로 분리합니다."""
    # 정답인 'sst'와 불필요한 컬럼을 제외한 모든 변수를 입력 데이터 X로 사용합니다.
    X = df.drop(columns=['sst', 'number', 'latitude', 'longitude', 'expver'])
    # 예측 목표인 'sst'를 정답 데이터 y로 사용합니다.
    y = df['sst']
    return X, y

X_train, y_train = split_X_y(train_featured)
X_validation, y_validation = split_X_y(validation_featured)
X_test, y_test = split_X_y(test_featured)


# --- LightGBM 모델 학습 ---
# PPT: "Model Training" 파트 핵심 로직
# LightGBM 회귀(Regressor) 모델을 정의합니다. random_state는 재현성을 위해 설정합니다.
model = lgb.LGBMRegressor(random_state=42)

print("===== LightGBM 모델 학습을 시작합니다... =====")
# 준비된 학습 데이터(X_train, y_train)를 사용하여 모델을 훈련시킵니다.
model.fit(X_train, y_train)
print("✅ 모델 학습 완료!")


# --- 모델 성능 평가 및 결과 시각화 ---
print("\n===== [테스트 데이터] 최종 성능 평가 결과 =====")
test_predictions = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f"최종 평균 절대 오차 (MAE): {test_mae:.4f} °C")
print(f"최종 제곱근 평균 제곱 오차 (RMSE): {test_rmse:.4f} °C")

# 1. 테스트 데이터 예측 결과 시계열 그래프
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(y_test.index, y_test, label='Actual SST (실제값)')
ax.plot(y_test.index, test_predictions, label='Predicted SST (예측값)', linestyle='--', alpha=0.7)
ax.set_title('Final SST Prediction vs Actual (Test Set)', fontsize=16)
ax.legend()
plt.show()

# 2. 특성 중요도 그래프
lgb.plot_importance(model, figsize=(10, 8), max_num_features=15)
plt.title('Feature Importance', fontsize=16)
plt.show()

# 3. 오차 분포 히스토그램
errors = y_test - test_predictions
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.title('Error Distribution (Test Set)', fontsize=16)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.show()

# 4. 예측-실제 산점도
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.5, s=10)
plt.title('Actual vs. Predicted SST (Test Set)', fontsize=16)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.grid(True)
plt.show()

# 5. 월별 오차 박스플롯
monthly_errors = pd.DataFrame({'month': y_test.index.month, 'error': errors})
plt.figure(figsize=(12, 7))
monthly_errors.boxplot(by='month', column='error', grid=False)
plt.title('Monthly Prediction Error Distribution (Test Set)', fontsize=16)
plt.suptitle('')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.show()
