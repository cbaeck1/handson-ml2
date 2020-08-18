import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image, housingModule
 
housing = housingModule.load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 예측 변수와 타깃 값에 같은 변형을 적용하지 않기 위해 예측 변수와 레이블을 분리
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity", axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
 
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values = np.nan, strategy = 'median')),
        ('attribs_adder', housingModule.CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
 
housing_num_tr = num_pipeline.fit_transform(housing_num)

# 
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
        ('selector', housingModule.DataFrameSelector(num_attribs)),
        # ('imputer', Imputer(strategy="median")),
        ('imputer', SimpleImputer(missing_values = np.nan, strategy = 'median')),
        ('attribs_adder', housingModule.CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
cat_pipeline = Pipeline([
        ('selector', housingModule.DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder()),
    ])

# SciPy 희소 행렬 sparse matrix
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))    

# 이 두 파이프라인을 하나의 파이프라인
from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape, type(housing_prepared))

# 훈련 세트에서 훈련하고 평가하기
# 선형 회귀 모델
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 5건으로 
some_data = housing.iloc[:5]
print("some_data:\n", some_data)
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측: ", lin_reg.predict(some_data_prepared))

# 사이킷런의 mean_square_error 함수를 사용해 전체 훈련 세트에 대한 이 회귀 모델의 RMSE를 측정
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("선형 회귀 모델 RMSE:", lin_rmse)

# 결정 트리
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("결정 트리 모델 RMSE:", tree_rmse)

# 교차 검증을 사용한 평가
# K-겹 교차 검증 K-fold cross-validation을 수행합니다. 
# 훈련 세트를 폴드 fold라 불리는 10개의 서브셋으로 무작위로 분할합니다. 
# 그런 다음 결정 트리 모델을 10번 훈련하고 평가하는데, 
# 매번 다른 폴드를 선택해 평가에 사용하고 나머지 9개 폴드는 훈련에 사용합니다.
# 10개의 평가 점수가 담긴 배열이 결과임

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

# 사이킷런의 교차 검증 기능은 scoring 매개변수에 (낮을수록 좋은) 비용 함수가 아니라 (클수록 좋은) 효용 함수를 기대합니다. 
# 그래서 평균 제곱 오차(MSE)의 반댓값(즉, 음숫값)을 계산하는 neg_mean_squared_error 함수를 사용합니다. 
# 이런 이유로 앞선 코드에서 제곱근을 계산하기 전에 -scores로 부호를 바꿨습니다

def display_scores(scores):
    print("교차 검증 Scores:", scores)
    print("교차 검증 Mean:", scores.mean())
    print("교차 검증 Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# 앙상블 학습 : 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
print(forest_rmse)

# 그리드 탐색
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)






