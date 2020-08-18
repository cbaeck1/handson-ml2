import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image, housingModule

housing = housingModule.load_housing_data()
print("housing.head", housing.head())
print("housing.info", housing.info())
print("type", type(housing)) # DataFrame
print("housing.describe", housing.describe())

# 전문가가 중간 소득이 중간 주택 가격을 예측하는 데 매우 중요하다고 이야기해주었다고 가정
# 중간 소득이 연속적인 숫자형 특성이므로 소득에 대한 카테고리 특성을 만들어야 합니다
# 소득 대부분은 $20,000~$50,000 사이에 모여 있지만 일부는 $60,000를 넘기도 합니다. 
# 계층별로 데이터셋에 충분한 샘플 수가 있어야 합니다. 
# 그렇지 않으면 계층의 중요도를 추정하는 데 편향이 발생
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

housing["income_cat"].hist(bins=10, figsize=(5,5))
plt.title("housing income_cat Histogram Plot")
image.save_fig("housing_income_cat_histogram_plot")   
plt.show()

# 소득 카테고리를 기반으로 계층 샘플링을 할 준비가 되었습니다. 
# 사이킷런의 StratifiedShuffleSplit를 사용할 수 있습니다.
from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 전체 주택 데이터셋에서 소득 카테고리의 비율
print(housing["income_cat"].value_counts() / len(housing))

# income_cat 특성을 삭제해서 데이터를 원래 상태로
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
#
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.title("데이터의 지리적인 산점도")
image.save_fig("housing_longitude_latitude_scatter")   
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.title("밀집된 지역이 잘 부각된 산점도")
image.save_fig("housing_longitude_latitude_scatter1")   
plt.show()

# 원의 반지름은 구역의 인구
# 색깔은 가격
# 미리 정의된 컬러 맵 color map 중 파란색(낮은 가격)에서 빨간색(높은 가격)까지 범위를 가지는 jet
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
plt.title("캘리포니아 주택 가격")
image.save_fig("housing_longitude_latitude_scatter2")   
plt.show()

# 상관관계 조사
# 표준 상관계수 standard correlation coefficient(피어슨의 r)
corr_matrix = housing.corr()
# 중간 주택 가격과 다른 특성 사이의 상관관계 크기
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 상관관계의 범위는 -1부터 1까지입니다. 1에 가까우면 강한 양의 상관관계를 가진다는 뜻
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.title("산점도 행렬")
image.save_fig("housing_correlations")   
plt.show()

# 
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.title("중간 소득 대 중간 주택 가격")
image.save_fig("housing_median_income_median_house_value_scatter")   
plt.show()

# 특성 조합으로 실험
# 머신러닝 알고리즘에 주입하기 전에 정제해야 할 조금 이상한 데이터를 확인
# 어떤 특성은 꼬리가 두꺼운 분포라서 데이터를 변형해야 할 것입니다(예를 들면 로그 스케일로)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "rooms_per_household", "bedrooms_per_room", "population_per_household"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.title("산점도 행렬")
image.save_fig("housing_add_features_correlations")   
plt.show()







