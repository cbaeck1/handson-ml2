import numpy as np

# zeros, ones 함수를 통해 0행렬, 또는 모든 행렬 값이 1인 행렬을 만들 수 있다.
# zeros 함수는 0으로 채워진 배열
x = np.zeros(5)
print(x)

# 3*3 배열 
x1 = np.ones((3,3))
x2 = np.zeros((3,3))
print(x1)
print(x2)

# empty 함수는 행렬 값이 초기화되지 않은 행렬을 생성
x3 = np.empty((3,3))
print(x3)

# arange 함수를 통해 범위 내의 값을 순차적으로 가지는 배열을 생성
x4 = np.arange(15)

print(x4, x4.dtype)

# ndim,shape, dtype 함수를 통해 차수와 배열 모양, 배열의 자료 타입을 확인
x4.dtype

# astype 함수를 통해 자료 타입의 변환이 가능
# 문자열 형태를 정수형으로 변환하는 것도 가능
float_x2 = x2.astype(np.float64)
print("---------------")
# 축(axis) : 차원
# 랭크(rank) : 축의 갯수
# 예) x1 = np.ones((3,4)) 3*4 행렬의 rank = 2
#    첫번째 축의 길이는 3 이고 두번째 축의 길이는 4
# 배열의 크기(shape) : 배열의 축 길이
# 배열의 size : 전체 원소의 갯수 3*4 = 12

a = np.zeros((3,4))
print(a.shape)
print(a.ndim)
print(a.size)
print(type(a))

np.full((3,4), np.pi)
# ndarray는 기존 파이썬과는 다르게 오직 같은 종류의 데이터만을 배열
data = [1,2,3,4,5]
arr = np.array(data)
print("data", data)
print("arr:{0}".format(arr))

data = [[1, 2, 3], [4, 5, 6]]
print("data", data)
x = np.array(data)
print("x:{0}".format(x))

# 파이썬의 기본 range 함수와 비슷
# array([1,2,3,4])
np.arange(1,5)
# array([1.,2.,3.,4.])
np.arange(1.0,5.0)
# array([1.,1.5,2.,2.5,3.,3.5,4.,4.5])
np.arange(1, 5, 0.5)

# 부동 소수를 사용하면 원소의 개수가 일정하지 않을 수 있습니
print(np.arange(0, 5/3, 1/3)) # 부동 소수 오차 때문에, 최댓값은 4/3 또는 5/3이 됩니다.
print(np.arange(0, 5/3, 0.333333333))
print(np.arange(0, 5/3, 0.333333334))

# 부동 소수를 사용할 땐 arange 대신에 linspace 함수를 사용
print(np.linspace(0, 5/3, 6))

#  (균등 분포인) 0과 1사이의 랜덤한 부동 소수로 3 \times 4 행렬을 초기화
np.random.rand(3,4)

# 평균이 0이고 분산이 1인 일변량 정규 분포(가우시안 분포)에서 샘플링한 랜덤한 부동 소수를 담은 3 \times 4 행렬
a = np.random.randn(3,4)

# dtype 속성으로 쉽게 데이터 타입을 확인
print(a.dtype, a)

# dtype 매개변수를 사용해서 배열을 만들 때 명시적으로 지정
# 가능한 데이터 타입 int8, int16, int32, int64, uint8|16|32|64, float16|32|64, complex64|128
d = np.arange(1, 5, dtype=np.complex64)
print(d.dtype, d)

# 아이템의 크기(바이트)를 반환
e = np.arange(1, 5, dtype=np.complex64)
print(e.itemsize, e)

g = np.arange(24)
print(g)
print("rank:", g.ndim)
g.shape = (6, 4)
print(g)
print("rank:", g.ndim)
# 2*3*4 = 24
g.shape = (2, 3, 4)
print(g)
print("rank:", g.ndim)

# 배열 크기 변경 4*6 = 24
g2 = g.reshape(4,6)
print(g2)
print("rank:", g2.ndim)

# 1행2열의 원소를 999 로
g2[1, 2] = 999

# 이에 상응하는 g의 원소도 수정
print(g)

# 동일한 데이터를 가리키는 새로운 1차원 ndarray
g3 = g.ravel()
print(g3)

# 산술 연산
# 산술 연산자(+, -, *, /, //, ** 등)는 모두 ndarray와 사용

a = np.array([14, 23, 32, 41])
b = np.array([5, 4, 3, 2])
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a // b =", a // b)
print("a % b =", a % b)
print("a ** b =", a ** b)

# 일반적으로 넘파이는 동일한 크기의 배열을 기대합니다. 그렇지 않은 상황에는 브로드캐시틍 규칙을 적용
# 규칙1 : 배열의 랭크가 동일하지 않으면 랭크가 맞을 때까지 랭크가 작은 배열 앞에 1을 추가
print(np.arange(5))
# (0,0,1) (0,0,2) (0,0,3) (0,0,4) (0,0,5)
h = np.arange(5).reshape(1, 1, 5)
print(h)
# 다음과 동일합니다: h + [[[10, 20, 30, 40, 50]]]
h2 = h + [10, 20, 30, 40, 50] 
print(h2)

# 규칙2 : 특정 차원이 1인 배열은 그 차원에서 크기가 가장 큰 배열의 크기에 맞춰 동작
k = np.arange(6).reshape(2, 3)
print(k)
# 다음과 같습니다: k + [[100, 100, 100], [200, 200, 200]]
k2 = k + [[100], [200]] 
print(k2)

# 규칙3 : 규칙 1 & 2을 적용했을 때 모든 배열의 크기가 같아야 함
# 업캐스팅 : dtype이 다른 배열을 합칠 때 넘파이는 (실제 값에 상관없이) 모든 값을 다룰 수 있는 타입으로 업캐스팅합니다.
k1 = np.arange(0, 5, dtype=np.uint8)
print(k1.dtype, k1)
# 모든 int8과 uint8 값(-128에서 255까지)을 표현하기 위해 int16이 필요
k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)
k3 = k1 + 1.5
print(k3.dtype, k3)


print(np.arange(10))
# (0,0,1) (0,0,2) (0,0,3) (0,0,4) (0,0,5)
# (0,1,1) (0,1,2) (0,1,3) (0,1,4) (0,1,5)
h = np.arange(10).reshape(1, 2, 5)
print(h)

print(np.arange(10))
# (0,1) (0,2) (0,3) (0,4) (0,5)
# (1,1) (1,2) (1,3) (1,4) (1,5)
h = np.arange(10).reshape(2, 5)
print(h)

# 조건 연산자 : 원소별로 적용
m = np.array([20, -5, 30, 40])

# 배열 인덱싱
a = np.array([1, 5, 3, 19, 13, 7, 3])
# 19
print(a[3])
# array([3, 19, 13])
print(a[2:5])
# array([3, 19, 13, 7])
print(a[2:-1])
# array([1, 5])
print(a[:2])
# array([3, 13, 3])
print(a[2::2])
# array([3, 7, 13, 19, 3, 5, 1])
print(a[::-1])

# 인데싱으로 수정
a[3]=999
# 슬라이싱을 사용해 ndarray를 수정
a[2:5] = [997, 998, 999]

# 보통의 파이썬 배열과 차이점
# 보통의 파이썬 배열과 대조적으로 ndarray 슬라이싱에 하나의 값을 할당하면 슬라이싱 전체에 복사됩니다. 브로드캐스팅
a[2:5] = -1
# array([1, 5, -1, -1, -1, 7, 3])
print(a)
# 중요한 점은 ndarray의 슬라이싱은 같은 데이터 버퍼를 바라보는 뷰(view)입니다. 
# 슬라이싱된 객체를 수정하면 실제 원본 ndarray가 수정
a_slice = a[2:6]
a_slice[1] = 1000
# array([ 1, 5, -1, 1000, -1, 7, 3])
print(a)
# 원본배열을 수정하면 슬라이싱 객체에도 반영
a[3] = 2000
# array([ -1, 2000, -1, 7])
print(a_slice)

# 데이터를 복사하려면 copy 메서드를 사용해야 합니다:
another_slice = a[2:6].copy()
another_slice[1] = 3000
# 원본 배열이 수정되지 않습니다
# array([ 1, 5, -1, 2000, -1, 7, 3])
print(a)


# 다차원 배열
# 다차원 배열은 비슷한 방식으로 각 축을 따라 인덱싱 또는 슬라이싱해서 사용합니다. 콤마로 구분합니다:
# 4행 12열
b = np.arange(48).reshape(4, 12)  
# array([[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
#        [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
#        [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]])
print(b)

b[1, 2] # 행 1, 열 2
b[1, :] # 행 1, 모든 열
b[1, :]
















