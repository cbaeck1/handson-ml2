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
print("housing_prepared.shape: {}".format(housing_prepared.shape))
print("housing_labels.shape: {}".format(housing_labels.shape))
print("housing_prepared 타입: {}".format(type(housing_prepared)))
print("housing_labels 타입: {}".format(type(housing_labels)))

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
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("랜덤 포레스트 RMSE:", forest_rmse)

# 그리드 탐색
# 탐색하고자 하는 하이퍼파라미터와 시도해볼 값을 지정
# 가능한 모든 하이퍼파라미터 조합에 대해 교차 검증을 사용해 평가
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

# 더 세밀하게 탐색하려면 위 예제의 n_estimators 하이퍼파라미터처럼 더 작은 값을 지정
# param_grid 설정에 따라 사이킷런이 
# 먼저 첫 번째 dict에 있는 n_estimators와 max_features 하이퍼파라미터의 조합인 3 × 4 = 12개를 평가
# 두 번째 dict에 있는 하이퍼파라미터의 조합인 2 × 3 = 6개를 시도
# 모두 합하면 그리드 탐색이 RandomForestRegressor 하이퍼파라미터 값의 12 + 6 = 18 개 조합을 탐색하고,
# 각각 다섯 번 모델을 훈련시킵니다 (5-겹 교차 검증을 사용하기 때문에).
# 다시 말해 전체 훈련 횟수는 18 × 5 = 90 이 됩니다! 
# 이는 시간이 꽤 오래 걸리지만 다음과 같이 최적의 조합을 얻을 수 있습니다.
print(grid_search.best_params_, grid_search.best_estimator_)

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
        max_features=8, max_leaf_nodes=None, min_impurity_split=1e-07,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
        oob_score=False, random_state=42, verbose=0, warm_start=False)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# 이 예에서는 max_features 하이퍼파라미터가 8, n_estimators 하이퍼파라미터가 30일 때 최적의 솔루션입니다. 
# 이때 RMSE 점수가 49,694로 앞서 기본 하이퍼파라미터 설정으로 얻은 52,564점보다 조금 더 좋습니다


# 랜덤 탐색
# 앙상블 방법
# 최상의 모델과 오차 분석
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# 테스트 세트로 시스템 평가하기
final_model = grid_search.best_estimator_
 
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
 
X_test_prepared = full_pipeline.transform(X_test)
 
final_predictions = final_model.predict(X_test_prepared)
 
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => 47,766.0 출력








