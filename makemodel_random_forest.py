# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터셋으로 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# 결과를 표 형식으로 변환
report_df = pd.DataFrame(report).transpose()

# 정확도 추가
accuracy_df = pd.DataFrame({'accuracy': [accuracy]})

# 결과 출력
print("Model Performance Report:")
print(report_df)
print("\nAccuracy:")
print(accuracy_df)

# 모델을 pkl 파일로 저장
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)