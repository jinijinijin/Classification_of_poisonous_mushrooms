import pandas as pd
import warnings

from sklearn.metrics import r2_score, accuracy_score

warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier


# 데이터 읽어 들이기
input_mushrooms = pd.read_csv('input.csv')
target_mushrooms = pd.read_excel('target.xlsx')
mushrooms = pd.read_csv('encoded_mushrooms.csv')

# 데이터셋 전처리
input_mushrooms = pd.get_dummies(input_mushrooms)
target_mushrooms = pd.get_dummies(target_mushrooms)
mushrooms = pd.get_dummies(mushrooms)

#상위 5개 데이터
print(input_mushrooms.head())
print(target_mushrooms.head())

#dataframe 모양 (행, 열)
print(input_mushrooms.shape)
print(target_mushrooms.shape)

#열 가져오기
data_input = input_mushrooms[['bruises', 'odor', 'gill-size', 'gill-color', 'ring-type']].to_numpy()
print(data_input[:5])

data_target =target_mushrooms['class'].to_numpy()



# 학습데이터와 테스트 데이터를 나누기 3:1
train_input, test_input, train_target, test_target = train_test_split(
    input_mushrooms, target_mushrooms,test_size=0.2, random_state=42)


from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

print("-----------")
print("train score")
print(knr.score(train_input,train_target))

#예측이 정확하면 1에 근접, 예측 = Target 평균 수준 => 0
print("-----------")
print("test score")
print(knr.score(test_input, test_target))

print("-----------")
print("과소적합")


print("------절댓값의 차이의 평균------")

from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input)
mae =mean_absolute_error(test_target, test_prediction)
print(mae)

#이웃 개수의 조정
knr.n_neighbors = 57
knr.fit(train_input, train_target)

print("<K-최근접 이웃회귀>")
print("-----------")
print("train score")
print(knr.score(train_input,train_target))
print("-----------")
print("test score")
print(knr.score(test_input,test_target))



print("==============>이웃수 증가할 수록 감소 : 주변부 특징에 민감==============")
print("=test해본 결과 5일땐 과소 90%대 점수 /200일땐 과대 70%대점수/ 4000했다가 0.1%......==")

#Linear
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)

print("선형회귀")
print("-----------")
print("train score")
print(lr.score(train_input, train_target))
print("-----------")
print("test score")
print(lr.score(test_input, test_target))

#다항회귀

train_poly = np.column_stack((train_input ** 5, train_input))
test_poly = np.column_stack((test_input ** 5, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)

print("<다항회귀>")
print("-----------")
print("train score")
print(lr.score(train_poly,train_target))
print("-----------")
print("test score")
print(lr.score(test_poly, test_target))

#다중회귀

from sklearn.preprocessing import PolynomialFeatures

#degree=2
poly = PolynomialFeatures()
poly.fit([[2,3]])

print((poly.transform([[2,3]])))

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(f'degree=2: {train_poly.shape}')

poly.get_feature_names_out()

test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)

print("<다중회귀>")
print("-----------")
print("train score")
print(lr.score(train_poly, train_target))
print("-----------")
print("test score")
print(lr.score(test_poly, test_target))

#3차항 까지 추가한다면??
poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)



print(f'3차항: {train_poly.shape}')
lr.fit(train_poly, train_target)

print("<3차항>")
print("-----------")
print("train score")
print(lr.score(train_poly, train_target))
print("-----------")
print("test score")
print(lr.score(test_poly, test_target))

from sklearn.linear_model import Ridge, Lasso, LogisticRegression

# 함수 차수 설정
poly = PolynomialFeatures(degree=3)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)
ss = StandardScaler()
ss.fit_transform(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# ridge 회귀 모델
ridge = Ridge(alpha=0.5)
ridge.fit(train_scaled, train_target)

print('Ridge Training score:', ridge.score(train_scaled, train_target))
print('Ridge Testing score:',ridge.score(test_scaled, test_target))

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 회귀 학습
    ridge = Ridge(alpha = alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# 그래프로 확인
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('ridge alpha')
plt.ylabel('R^2')
plt.show()

# Lasso 회귀 모델
lasso = Lasso()
lasso.fit(train_scaled, train_target)

train_score = []
test_score = []

print('Lasso Training score:', lasso.score(train_scaled, train_target))
print('Lasso Testing score:',lasso.score(test_scaled, test_target))

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 회귀 학습
    lasso = Lasso(alpha = alpha, max_iter = 10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# 그래프로 확인
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)

plt.xlabel('lasso alpha')
plt.ylabel('R^2')
plt.show()

lasso = Lasso(alpha=0.001)
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
print(np.sum(lasso.coef_ == 0))


# 로지스틱 회귀 모델 생성
lr = LogisticRegression(C=2)

# 모델 학습
lr.fit(train_scaled, train_target)

print('Logistic Regression Train score:', lr.score(train_scaled, train_target))
print('Logistic Regression Test score:', lr.score(test_scaled, test_target))

#<SGD>
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
train_score = []
test_score = []

#적합 ㄱㅊ
print(f'SGD 훈련 세트 점수: {sc.score(train_scaled, train_target)}')
print(f'SGD 테스트 세트 점수: {sc.score(test_scaled, test_target)}')
#적합 ㄱㅊ

sc.partial_fit(train_scaled, train_target)
print(f'SGD epoch 1회 추가한 훈련 세트 점수: {sc.score(train_scaled, train_target)}')
print(f'SGD epoch 1회 추가한 테스트 세트 점수: {sc.score(test_scaled, test_target)}')

classes = np.unique(train_target)
#조기종료
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss='log', max_iter =100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

#적합 ㄱㅊ
print(f'SGD 최적의 epoch 훈련 세트 점수: {sc.score(train_scaled, train_target)}')
print(f'SGD 최적의 epoch 테스트 세트 점수: {sc.score(test_scaled, test_target)}')

#<결정트리>
from sklearn.tree import DecisionTreeClassifier

#결정트리
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(train_input, train_target)

#과소적합 0.001퍼정도 테스트가 높음
print(f'depth = 5 결정 트리 훈련 세트 점수: {dt.score(train_input, train_target)}')
print(f'depth = 5 결정 트리 테스트 세트 점수: {dt.score(test_input, test_target)}')

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#결정트리 분석
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['bruises','odor','gill-size','gill-color','stalk-surface-above-ring','ring-type','spore-print-color'])
plt.show()
print(dt.feature_importances_)
