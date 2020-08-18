import pandas as pd
import numpy as np
import mglearn, os

import matplotlib as mpl
import matplotlib.pyplot as plt
import image, housingModule
 
housing = housingModule.load_housing_data()
print("housing.head", housing.head())
print("housing.info", housing.info())
print("housing.describe", housing.describe())

# 숫자형 특성을 히스토그램
# 히스토그램은 주어진 값의 범위(수평축)에 속한 샘플 수(수직축)
# 중간 소득 median income
# 중간 주택 연도 housing median age
# 중간 주택 가격 median house value
housing.hist(bins=50, figsize=(20,15))
plt.title("모든 숫자형 특성에 대한 히스토그램")
image.save_fig("housing_histogram_plot")   
plt.show()

# 테스트 세트 만들기
# 데이터 스누핑 data snooping 편향
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# 프로그램을 다시 실행하면 다른 테스트 세트가 생성
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

# 항상 같은 난수 인덱스가 생성되도록 np.random.permutation() 을 호출하기 전에 난수 발생기의 초기값을 지정
# 식별자의 해시값을 계산하여 해시의 마지막 바이트의 값이 51(256의 20% 정도)보다 작거나 같은 샘플만 테스트 세트
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# ‘index’ 열이 추가된 데이터프레임이 반환됩니다.
housing_with_id = housing.reset_index() 
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# 사이킷런은 데이터셋을 여러 서브셋으로 나누는 다양한 방법을 제공
from sklearn.model_selection import train_test_split
 
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# 미국 인구의 51.3%가 여성이고 48.7%가 남성이라면, 잘 구성된 설문조사는 샘플에서도 이 비율을 유지해야 합니다. 
# 즉, 여성은 513명, 남성은 487명이어야 합니다. 이를 계층적 샘플링 stratified sampling 이라고 합니다. 
# 전체 모수는 계층strata이라는 동질의 그룹으로 나뉘고, 테스트 세트가 전체 모수를 대표하도록 
# 각 계층에서 올바른 수의 샘플을 추출

# 전문가가 중간 소득이 중간 주택 가격을 예측하는 데 매우 중요하다고 이야기해주었다고 가정
# 중간 소득이 연속적인 숫자형 특성이므로 소득에 대한 카테고리 특성을 만들어야 합니다
# 소득 대부분은 $20,000~$50,000 사이에 모여 있지만 일부는 $60,000를 넘기도 합니다. 
# 계층별로 데이터셋에 충분한 샘플 수가 있어야 합니다. 
# 그렇지 않으면 계층의 중요도를 추정하는 데 편향이 발생

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
print(housing["income_cat"].value_counts() / len(housing))

housing["income_cat"].hist(bins=10, figsize=(5,5))
plt.title("소득 카테고리의 히스토그램")
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



    