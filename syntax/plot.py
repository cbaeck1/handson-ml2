import matplotlib.pyplot as plt
import numpy as np

# 입력 데이터
num = 10
X = np.arange(num)
W = np.random.randint(1, num*2, num)

# plot 입력
hist = plt.hist(X, bins=num, weights=W, density=False, cumulative=False, label='A',
                range=(X.min()-1, X.max()+1), color='r', edgecolor='black', linewidth=1.2)

# 그래프의 타이틀과 x,y축 라벨링
plt.title('scatter', pad=10)
plt.xlabel('X axis', labelpad=10)
plt.ylabel('Y axis', labelpad=10)

# 틱설정
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', pad=8, top=True, right=True)

# 플롯 출력
plt.show()


