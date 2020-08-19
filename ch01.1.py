import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import image, housingModule

# 데이터 적재
oecd_bli = housingModule.load_oecd_bli_data()
print("oecd_bli 크기: {}".format(oecd_bli.shape))
gdp_per_capita = housingModule.load_gdp_per_capita_data()
print("gdp_per_capita 크기: {}".format(gdp_per_capita.shape))

# 데이터 준비
country_stats = housingModule.prepare_country_stats(oecd_bli, gdp_per_capita)
print("country_stats 크기: {}".format(country_stats.shape))

# X = np.c_[country_stats["GDP per capita"]]
X = country_stats[["GDP per capita"]]
y = pd.Series(country_stats["Life satisfaction"])
print("X 크기: {}".format(X.shape))
print("y 크기: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))

# 데이터 시각화
ax = country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
ax.set(xlabel="1인당 GDP", ylabel="삶의 만족도")
plt.title("삶의만족도와 1인당GDP 상관도")
image.save_fig("life_satisfaction_gdp_per_capita_scatter")   
plt.show()

# 선형 모델 선택
# X : numpy 배열 또는 형태의 희소 행렬 [n_samples, n_features]
from sklearn.linear_model import LinearRegression 
lm = LinearRegression().fit(X, y)
# 모델 훈련
# Exception has occurred: ValueError illegal value in 4-th argument of internal None
print("lm.coef_: {}".format(lm.coef_))
print("lm.intercept_: {}".format(lm.intercept_))
print("2. 선형모델 : 최소제곱 훈련 세트 점수: {:.2f}".format(lm.score(X, y)))

# 키프로스에 대한 예측
X_new = [[22587]]  # 키프로스 1인당 GDP
print(lm.predict(X_new)) # 결과 [[5.96242338]]

# 선형 회귀 모델을 k-최근접 이웃 회귀 모델로 교체할 경우
import sklearn.neighbors
knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# 모델 훈련
knn.fit(X, y)
# 키프로스에 대한 예측
print(knn.predict(X_new)) # 결과 [[5.76666667]]

# 
oecd_bli = housingModule.load_oecd_bli_data()
gdp_per_capita = housingModule.load_gdp_per_capita_data()
full_country_stats = housingModule.prepare_full_country_stats(oecd_bli, gdp_per_capita)
full_country_stats.sort_values(by="GDP per capita", inplace=True)
print("full_country_stats 크기: {}".format(full_country_stats.shape))
print(full_country_stats[["GDP per capita", 'Life satisfaction']].loc["United States"])
remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))

sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

#
ax = sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary": (5000, 1, '헝가리'),
    "Korea": (18000, 1.7, '대한민국'),
    "France": (29000, 2.4, '프랑스'),
    "Australia": (40000, 3.0, '호주'),
    "United States": (52000, 3.8, '미국'),
}
for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = sample_data.loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(pos_text[2], xy=(pos_data_x, pos_data_y), xytext=pos_text[:2],
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")
plt.title("삶의만족도와 1인당GDP 상관도")    
image.save_fig('money_happy_scatterplot')
plt.show()


sample_data.to_csv(os.path.join("datasets", "lifesat", "lifesat.csv"))
sample_data.loc[list(position_text.keys())]

ax = sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
plt.axis([0, 60000, 0, 10])
X=np.linspace(0, 60000, 1000)
plt.plot(X, 2*X/100000, "r")
plt.text(40000, 2.7, r"$\theta_0 = 0$", fontsize=14, color="r")
plt.text(40000, 1.8, r"$\theta_1 = 2 \times 10^{-5}$", fontsize=14, color="r")
plt.plot(X, 8 - 5*X/100000, "g")
plt.text(5000, 9.1, r"$\theta_0 = 8$", fontsize=14, color="g")
plt.text(5000, 8.2, r"$\theta_1 = -5 \times 10^{-5}$", fontsize=14, color="g")
plt.plot(X, 4 + 5*X/100000, "b")
plt.text(5000, 3.5, r"$\theta_0 = 4$", fontsize=14, color="b")
plt.text(5000, 2.6, r"$\theta_1 = 5 \times 10^{-5}$", fontsize=14, color="b")
image.save_fig('tweaking_model_params_plot')
plt.show()

# lin1 = LinearRegression()
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
# lin1.fit(Xsample, ysample)
# t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
# print(t0, t1)


ax = sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
plt.axis([0, 60000, 0, 10])
X=np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1*X, "b")
plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
save_fig('best_fit_model_plot')
plt.show()


cyprus_gdp_per_capita = gdp_per_capita.loc["Cyprus"]["GDP per capita"]
print(cyprus_gdp_per_capita)
cyprus_predicted_life_satisfaction = lin1.predict(cyprus_gdp_per_capita.reshape(-1,1))[0][0]
cyprus_predicted_life_satisfaction


ax = sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3), s=1)
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
X=np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1*X, "b")
plt.axis([0, 60000, 0, 10])
plt.text(5000, 7.5, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 6.6, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita], [0, cyprus_predicted_life_satisfaction], "r--")
plt.text(25000, 5.0, r"예측 = 5.96", fontsize=14, color="b")
plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")
save_fig('cyprus_prediction_plot')
plt.show()

print(sample_data[7:10])
print(missing_data)


position_text2 = {
    "Brazil": (1000, 9.0, '브라질'),
    "Mexico": (11000, 9.0, '멕시코'),
    "Chile": (25000, 9.0, '칠레'),
    "Czech Republic": (35000, 9.0, '체코'),
    "Norway": (60000, 3, '노르웨이'),
    "Switzerland": (72000, 3.0, '스위스'),
    "Luxembourg": (90000, 3.0, '룩셈부르크'),
}

ax = sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
plt.axis([0, 110000, 0, 10])

for country, pos_text in position_text2.items():
    pos_data_x, pos_data_y = missing_data.loc[country]
    plt.annotate(pos_text[2], xy=(pos_data_x, pos_data_y), xytext=pos_text[:2],
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")

X=np.linspace(0, 110000, 1000)
plt.plot(X, t0 + t1*X, "b:")
lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[full_country_stats["GDP per capita"]]
yfull = np.c_[full_country_stats["Life satisfaction"]]
lin_reg_full.fit(Xfull, yfull)
t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
X = np.linspace(0, 110000, 1000)
plt.plot(X, t0full + t1full * X, "k")
save_fig('representative_training_data_scatterplot')
plt.show()



ax = full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
ax.set(xlabel='1인당 GDP', ylabel='삶의 만족도')
plt.axis([0, 110000, 0, 10])

from sklearn import preprocessing
from sklearn import pipeline

poly = preprocessing.PolynomialFeatures(degree=60, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
save_fig('overfitting_model_plot')
plt.show()


plt.figure(figsize=(8,3))

plt.xlabel("1인당 GDP")
plt.ylabel('삶의 만족도')

plt.plot(list(sample_data["GDP per capita"]), list(sample_data["Life satisfaction"]), "bo")
plt.plot(list(missing_data["GDP per capita"]), list(missing_data["Life satisfaction"]), "rs")

X = np.linspace(0, 110000, 1000)
plt.plot(X, t0full + t1full * X, "r--", label="모든 데이터로 만든 선형 모델")
plt.plot(X, t0 + t1*X, "b:", label="일부 데이터로 만든 선형 모델")

ridge = linear_model.Ridge(alpha=10**9.5)
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
ridge.fit(Xsample, ysample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
plt.plot(X, t0ridge + t1ridge * X, "b", label="일부 데이터로 만든 규제가 적용된 선형 모델")

plt.legend(loc="lower right")
plt.axis([0, 110000, 0, 10])
save_fig('ridge_model_plot')
plt.show()







