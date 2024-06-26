# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 리스트
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 모델 학습 및 평가
for model_name, model in models.items():
    print(f"\nModel: {model_name}")

    # 모델 학습
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
    with open(f'{model_name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
        pickle.dump(model, file)