# 🌊 SST Analytics Dashboard

**한국해양과학기술원(KIOST) 해양위성센터(KOSC) 인턴십 프로젝트**

부산 연안 해수면 온도(SST)를 LightGBM 모델로 예측하고, Streamlit 웹 앱으로 시각화하는 프로젝트입니다.

**배포 주소**: https://sstpredictionapp-bgu6hfkkjvy4vnx5jtyqtl.streamlit.app/

## 주요 기능

- **SST 예측** - 특정 날짜/조건을 입력하면 해수면 온도를 예측합니다
- **시계열 분석** - 과거 데이터 기반 SST 변화 추이를 확인할 수 있습니다
- **시각화** - 실측값과 예측값 비교 그래프를 제공합니다
- **CSV 업로드** - 사용자 데이터를 업로드하여 일괄 예측이 가능합니다
- **Feature Importance** - 모델의 변수 중요도를 확인할 수 있습니다

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.8+ |
| 모델 | LightGBM |
| 웹 프레임워크 | Streamlit |
| 데이터 처리 | Pandas, NumPy |
| 시각화 | Matplotlib, Seaborn |

## 데이터

- **출처**: ECMWF ERA5 재분석 자료
- **형식**: CSV (train, validation, test)
- **주요 변수**: 기온, 해면기압, 풍속, 일사량 등
- **예측 대상**: 해수면 온도 (SST, ℃)

## 모델

- **알고리즘**: LightGBM (Gradient Boosting Decision Tree)
- **Feature Engineering**: SST, 기온, 풍속에 대한 1/2/3/7일 래그 변수 생성
- **평가 지표**: RMSE, MAE, R²

## 📁 프로젝트 구조

```
├── LightGBM_modeling.py      # 모델 학습 및 평가
├── LightGBM_app.py           # Streamlit 앱
├── requirements.txt          # 라이브러리 목록
├── train_data.csv
├── validation_data.csv
└── test_data.csv
```

## 실행 방법

```bash
git clone https://github.com/yangbeomseok/sst_analytics_dashboard.git
cd sst_analytics_dashboard
pip install -r requirements.txt
streamlit run LightGBM_app.py
```

## 📄 라이선스

MIT License
