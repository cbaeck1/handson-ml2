import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image, housingModule
 
housing = housingModule.load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
print("housing.info", housing.info())

from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 예측 변수와 타깃 값에 같은 변형을 적용하지 않기 위해 예측 변수와 레이블을 분리
housing = strat_train_set.drop("median_house_value", axis=1)
print("housing:\n", housing)
housing_labels = strat_train_set["median_house_value"].copy()
print("housing_labels:\n", housing_labels)

# 데이터 정제
# 1. 해당 구역을 제거합니다.
# 2. 전체 특성을 삭제합니다.
# 3. 어떤 값으로 채웁니다(0, 평균, 중간값 등).
housing.dropna(subset=["total_bedrooms"]) 
# 특성을 삭제
housing.drop("total_bedrooms", axis=1) 
# 중간값으로 
median = housing["total_bedrooms"].median() 
housing["total_bedrooms"].fillna(median, inplace=True)

# 사이킷런의 Imputer는 누락된 값을 손쉽게 다루도록 해줍니다. 
# 누락된 값을 특성의 중간값으로 대체한다고 지정하여 Imputer의 객체를 생성
# Imputer was deprecated 3 versions ago and remove in 0.22
# from sklearn.preprocessing import Imputer
# imputer = Imputer(strategy="median")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
housing_num = housing.drop("ocean_proximity", axis=1)
# imputer는 각 특성의 중간값을 계산해서 그 결과를 객체의 statistics_ 속성에 저장
imputer.fit(housing_num)
# print(imputer.statistics_)

# 훈련 세트에서 누락된 값을 학습한 중간값으로 대체
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = list(housing.index.values))

'''
사이킷런의 설계 철학
사이킷런의 API는 아주 잘 설계되어 있습니다. 주요 설계 원칙은 다음과 같습니다.
1. 일관성. 모든 객체가 일관되고 단순한 인터페이스를 공유합니다.
추정기estimator. 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체를 추정기라고 합니다
(예를 들어 imputer 객체는 추정기입니다). 추정 자체는 fit() 메서드에 의해 수행되고 하나의 매개변수로 
하나의 데이터셋만 전달합니다(지도 학습 알고리즘에서는 매개변수가 두 개로, 두 번째 데이터셋은 레이블을 담고 있습니다). 
추정 과정에서 필요한 다른 매개변수들은 모두 하이퍼파라미터로 간주되고(예를 들면 imputer 객체의 strategy 매개변수), 
인스턴스 변수로 저장됩니다(보통 생성자의 매개변수로 전달합니다).

2. 변환기transformer. (imputer 같이) 데이터셋을 변환하는 추정기를 변환기라고 합니다. 
여기서도 API는 매우 단순합니다. 변환은 데이터셋을 매개변수로 전달받은 transform() 메서드가 수행합니다. 
그리고 변환된 데이터셋을 반환합니다. 이런 변환은 일반적으로 imputer의 경우와 같이 학습된 모델 파라미터에 의해 결정됩니다.
모든 변환기는 fit()과 transform()을 연달아 호출하는 것과 동일한 fit_transform() 메서드도 가지고 있습니다
(이따금 fit_transform()이 최적화되어 있어서 더 빠릅니다).

3. 예측기predictor. 일부 추정기는 주어진 데이터셋에 대해 예측을 만들 수 있습니다. 
예를 들어 앞 장에 나온 LinearRegression 모델이 예측기입니다. 
어떤 나라의 1인당 GDP가 주어질 때 삶의 만족도를 예측했습니다. 
예측기의 predict() 메서드는 새로운 데이터셋을 받아 이에 상응하는 예측값을 반환합니다. 
또한 테스트 세트(지도 학습 알고리즘이라면 레이블도 함께)를 사용해 예측의 품질을 측정하는 score() 메서드를 가집니다.

4. 검사 가능. 모든 추정기의 하이퍼파라미터는 공개public 인스턴스 변수로 직접 접근할 수 있고
(예를 들면 imputer.strategy), 
모든 추정기의 학습된 모델 파라미터도 접미사로 밑줄을 붙여서 공개 인스턴스 변수로 제공됩니다
(예를 들면 imputer.statistics_).
클래스 남용 방지. 데이터셋을 별도의 클래스가 아니라 넘파이 배열이나 사이파이 희소sparse 행렬로 표현합니다. 
하이퍼파라미터는 보통의 파이썬 문자열이나 숫자입니다.

5. 조합성. 기존의 구성요소를 최대한 재사용합니다. 
앞으로 보겠지만 예를 들어 여러 변환기를 연결한 다음 마지막에 추정기 하나를 배치한 Pipeline 추정기를 쉽게 만들 수 있습니다.

6. 합리적인 기본값. 사이킷런은 일단 돌아가는 기본 시스템을 빠르게 만들 수 있도록 
대부분의 매개변수에 합리적인 기본값을 지정해두었습니다.
'''

# 텍스트와 범주형 특성 다루기 : 범주형 특성 ocean_proximity
housing_cat = housing["ocean_proximity"]
print("housing_cat:\n", housing_cat.head(10))

# 카테고리를 정숫값으로 매핑 : factorize()
housing_cat_encoded, housing_categories = housing_cat.factorize()
print(housing_cat_encoded[:10], housing_categories)

# 정수값을 명목형으로 순서가 의미가 없습니다. 의미를 가질려면
# 한 특성만 1이고(핫) 나머지는 0 으로 변환하는 이진 특성으로 변환
# 원-핫 인코딩 one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
# SciPy 희소 행렬 sparse matrix
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot, type(housing_cat_1hot))

# 넘파이 배열로
housing_cat_1hot = housing_cat_1hot.toarray()
print(housing_cat_1hot, type(housing_cat_1hot))

# cannot import name 'CategoricalEncoder' from 'sklearn.preprocessing' 
# pip install git+git://github.com/scikit-learn/scikit-learn.git
# CategoricalEncoder is only available in the development version 0.20.dev0. 
# Use OneHotEncoder and OrdinalEncoder insted.(see https://github.com/scikit-learn/scikit-learn/issues/10521 )
'''
from sklearn.preprocessing import CategoricalEncoder 
cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
print(housing_cat_1hot, type(housing_cat_1hot))

# 밀집 행렬 dense matrix 을 원할 경우에는 encoding 매개변수를 “onehot-dense”로 지정
cat_encoder = CategoricalEncoder(encoding="onehot-dense")
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
print(housing_cat_1hot, type(housing_cat_1hot))
'''

attr_adder = housingModule.CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# 특성 스케일링
# 전체 방 개수의 범위는 6 ~ 39,320
# 중간 소득의 범위는 0 ~ 15
# min-max 스케일링(정규화 normalization)과 표준화 standardization 가 널리 사용
# min-max 스케일링 : 데이터에서 최소값을 뺀 후 최대값과 최소값의 차이로 나눔, MinMaxScaler 
# 표준화 standardization : 데이터에서 평균을 뺀 후 표준편차로 나눔 (분산=1, 평균=0), StandardScaler 

# 변환 파이프라인
# 연속된 변환을 순서대로 처리할 수 있도록 도와주는 Pipeline 클래스
# Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
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
