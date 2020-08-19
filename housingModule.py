import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
 
DOWNLOAD_HOUSING = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_HOUSING + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

DOWNLOAD_BLI = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
OECD_BLI_PATH = os.path.join("datasets", "lifesat")
def fetch_bli_data(oecd_bli_path=OECD_BLI_PATH):
    if not os.path.isdir(oecd_bli_path):
        os.makedirs(oecd_bli_path, exist_ok=True)
    for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
        print("Downloading", filename)
        url = DOWNLOAD_BLI + "datasets/lifesat/" + filename
        urllib.request.urlretrieve(url, oecd_bli_path + filename)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def load_oecd_bli_data(oecd_bli_path=OECD_BLI_PATH):
    csv_path = os.path.join(oecd_bli_path, "oecd_bli_2015.csv")
    return pd.read_csv(csv_path, thousands=',')

def load_lifesat_data(oecd_bli_path=OECD_BLI_PATH):
    csv_path = os.path.join(oecd_bli_path, "lifesat.csv")
    return pd.read_csv(csv_path, thousands=',')

def load_gdp_per_capita_data(oecd_bli_path=OECD_BLI_PATH):
    csv_path = os.path.join(oecd_bli_path, "gdp_per_capita.csv")
    return pd.read_csv(csv_path, thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# OECD의 삶의 만족도(life satisfaction) 데이터와 IMF의 1인당 GDP(GDP per capita) 데이터
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# OECD의 삶의 만족도(life satisfaction) 데이터와 IMF의 1인당 GDP(GDP per capita) 데이터
def prepare_full_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    return full_country_stats

# 나만의 변환기
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # *args나 **kargs가 아닙니다.
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    def fit(self, X, y=None):
        return self # 더 할 일이 없습니다.
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household] 

# 사이킷런이 판다스의 데이터프레임을 다룰 수는 없지만 이를 처리하는 변환기를 직접 만들 수는 있습니다.
# 필요한 특성을 선택하여 데이터프레임을 넘파이 배열로 바꾸는 식으로 데이터를 변환 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


if __name__ == "__main__":
  # 최초 한번 실행
  # fetch_housing_data()
  housing = load_housing_data()
  print("housing.head", housing.head())
  print("housing.info", housing.info())
  print("housing.describe", housing.describe())

  oecd_bli = load_oecd_bli_data()
  print("oecd_bli.head", oecd_bli.head())
  print("oecd_bli.info", oecd_bli.info())
  print("oecd_bli.describe", oecd_bli.describe())

  lifesat = load_lifesat_data()
  print("lifesat.head", lifesat.head())
  print("lifesat.info", lifesat.info())
  print("lifesat.describe", lifesat.describe())

  gdp_per_capita = load_gdp_per_capita_data()
  print("gdp_per_capita.head", gdp_per_capita.head())
  print("gdp_per_capita.info", gdp_per_capita.info())
  print("gdp_per_capita.describe", gdp_per_capita.describe())

  country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
  print("country_stats.head", country_stats.head())
  print("country_stats.info", country_stats.info())
  print("country_stats.describe", country_stats.describe())    












