import pandas as pd
import pickle
import numpy as np
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

drive.mount('/content/drive')

file_path = '/content/drive/My Drive/synthetic_dataset/goal_set.p'
with open(file_path, 'rb') as file:
    consultation_data = pickle.load(file)

train_data = consultation_data['train']

disease_tags = []
symptoms = []

for item in train_data:
    disease_tags.append(item['disease_tag'])
    explicit_symptoms = list(item['goal']['explicit_inform_slots'].keys())
    implicit_symptoms = list(item['goal']['implicit_inform_slots'].keys())
    all_symptoms = explicit_symptoms + implicit_symptoms
    symptoms.append(" ".join(all_symptoms))

# DataFrame 생성
df = pd.DataFrame({
    'Disease Tag': disease_tags,
    'Symptoms': symptoms
})

# 데이터 분할
X = df['Symptoms']
y = df['Disease Tag']

# 레이블 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텍스트 데이터 벡터화
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 모델 정의
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# DCG, IDCG 및 NDCG 함수 정의
def dcg(rel, i):
    return rel / np.log2(i + 1)  # 주어진 순위에 대한 Discounted Cumulative Gain 계산

def idcg(rel, data_length):
    accum_idcg = 0
    for i in range(1, data_length + 1):
        accum_idcg += dcg(rel, i)  # 주어진 데이터 길이에 대한 Ideal DCG 계산
    return accum_idcg

def ndcg_k(proba, ground, k):
    ndcg_result = []
    target_score = 1

    top_k = np.flip(np.argsort(proba), axis=1)[:, :k]  # 상위 k개의 예측값 가져오기
    for y_h, y in zip(top_k, ground):
        try:
            len(y)
            label_type = 'multi'
        except Exception as e:
            label_type = 'single'

        if label_type == 'multi':
            accum_dcg = 0
            accum_idcg = idcg(target_score, len(y))
            for ea_y in y:
                if ea_y in y_h:
                    accum_dcg += dcg(target_score, np.where(y_h == ea_y)[0][0] + 1)
            if accum_dcg == 0 or accum_idcg == 0:
                ndcg_result.append(0)
            else:
                ndcg_result.append(accum_dcg / accum_idcg)

    return np.mean(ndcg_result) * 100  # 평균 NDCG 반환

# 각 모델을 학습하고 평가
for name, model in models.items():
    # 모델 학습
    model.fit(X_train, y_train)

    # 테스트 세트에 대한 예측 레이블
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # NDCG 계산을 위한 y_test 이진화
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    # 예측 점수 가져오기
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)

    # y_test_bin과 y_score가 동일한 형태인지 확인
    if y_test_bin.shape != y_score.shape:
        raise ValueError("형태 불일치: y_test_bin과 y_score는 동일한 형태여야 합니다.")

    # NDCG@5 계산
    ndcg = ndcg_k(y_score, y_test_bin, k=5)

    print(f"{name} 정확도: {accuracy:.2f}, NDCG@5: {ndcg:.5f}")

    # 추가 비교를 위한 NDCG@10 계산
    ndcg_at_10 = ndcg_k(y_score, y_test_bin, k=10)
    print(f"{name} NDCG@10: {ndcg_at_10:.5f}")

# 증상 일치 및 질병 예측 로직
user_input = input("증상과 일치하는 단어를 입력하세요: ").strip().lower()
matching_symptoms = []

for symptom in symptom_disease_map.keys():
    if user_input in symptom.lower():
        matching_symptoms.append(symptom)

# 일치하는 증상 출력
if matching_symptoms:
    print(f"'{user_input}'와 일치하는 증상:")
    for idx, symptom in enumerate(matching_symptoms, 1):
        print(f"{idx}. {symptom}")

    selection = input("가지고 있는 증상에 해당하는 번호를 입력하세요: ")

    if selection.isdigit() and 1 <= int(selection) <= len(matching_symptoms):
        user_input_symptom = matching_symptoms[int(selection) - 1]
        print(f"선택한 증상: {user_input_symptom}")
    else:
        print("잘못된 선택입니다. 증상에 해당하는 번호를 입력하세요.")
else:
    print(f"'{user_input}'와 일치하는 증상이 없습니다.")

if 'user_input_symptom' in locals():
    associated_diseases = symptom_disease_map.get(user_input_symptom, [])

    if associated_diseases:
        print(f"'{user_input_symptom}'와 관련된 질병:")
        for idx, disease in enumerate(associated_diseases, 1):
            print(f"{idx}. {disease}")

        all_associated_disease_symptoms = []

        for disease in associated_diseases:
            disease_data = df[df['Disease Tag'] == disease]

            if not disease_data.empty:
                disease_symptoms = disease_data['Symptoms'].iloc[0]
                all_associated_disease_symptoms.extend(disease_symptoms.split())

        all_associated_disease_symptoms = list(set(all_associated_disease_symptoms))

        if user_input_symptom in all_associated_disease_symptoms:
            all_associated_disease_symptoms.remove(user_input_symptom)

        total_associated_diseases = len(associated_diseases)

        print("\n발견된 질병과 관련된 모든 증상:")

        symptom_count = {symptom: 0 for symptom in all_associated_disease_symptoms}

        for disease in associated_diseases:
            for symptom in symptom_disease_map.get(disease, []):
                if symptom in symptom_count:
                    symptom_count[symptom] += 1

        sorted_symptoms = sorted(symptom_count.items(), key=lambda x: x[1])

        for idx, (symptom, probability) in enumerate(sorted_symptoms, 1):
            if symptom in symptom_count:
                print(f"{idx}. {symptom}: {probability:.5f}%")
    else:
        print(f"'{user_input_symptom}'와 관련된 질병이 없습니다.")
else:
    print("선택된 증상이 없습니다.")

# 질병이 발견되지 않은 경우 다른 증상 제안
diseases_found = False
encountered_diseases = set()
other_symptoms = set()

for symptom, probability in sorted_symptoms:
    if symptom == user_input_symptom:
        continue

    response = input(f"{symptom}가 있습니까? (yes/no): ").lower().strip()
    if response == 'yes':
        combined_symptoms = [user_input_symptom, symptom]

        combined_diseases = []
        for index, row in df.iterrows():
            disease_tag = row['Disease Tag']
            symptoms = row['Symptoms']
            if user_input_symptom in symptoms and symptom in symptoms:
                if disease_tag not in encountered_diseases:
                    combined_diseases.append(disease_tag)
                    encountered_diseases.add(disease_tag)
                other_symptoms.update(symptoms.split())

        if combined_diseases:
            print(f"'{user_input_symptom}'와 '{symptom}'을 포함하는 질병:")
            for idx, disease in enumerate(combined_diseases, 1):
                print(f"{idx}. {disease}")
            diseases_found = True
            break
        else:
            print(f"'{user_input_symptom}'와 '{symptom}'을 포함하는 질병이 없습니다.")
    elif response == 'no':
        continue
    else:
        print("잘못된 응답입니다. 'yes' 또는 'no'를 입력하세요.")

if diseases_found:
    print("\n관련 질병의 다른 증상들:")
    for idx, symptom in enumerate(other_symptoms, 1):
        print(f"{idx}. {symptom}")
