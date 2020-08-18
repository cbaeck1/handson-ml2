import os
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
# print('사용가능한 TTF 폰트 개수:', len(font_list))
# print(font_list) 

# for font in fm.fontManager.ttflist:
#     print(font.name, font.fname)    
#     if 'Nanum' in font.name:
#        print(font.name, font.fname)    

# 1) FontProperties 를 사용하는 방법 - 그래프의 폰트가 필요한 항목마다 지정해 주어야 합니다.
# 2) matplotlib.rcParams[]으로 전역글꼴 설정 방법 - 그래프에 설정을 해주면 폰트가 필요한 항목에 적용 됩니다.
# 3) 2)번의 방법을 mpl.matplotlib_fname()로 읽어지는 설정 파일에 직접 적어주는 방법, 단 모든 노트북에 적용됩니다. 노트북을 열 때마다 지정해 주지 않아도 돼서 편리합니다.

import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['axes.unicode_minus'] = False
# fname 옵션을 사용하는 방법
path = "c:/Windows/Fonts/malgun.ttf" # For Windows
# path = 'C:/Windows/Fonts/NanumBarunpenRegular.ttf'
font_name = fm.FontProperties(fname=path).get_name()
mpl.rc('font', family=font_name)

# 1) FontProperties 를 사용하는 방법
# fontprop = fm.FontProperties(fname=path, size=18)
# plt.title('시간별 가격 추이', fontproperties=fontprop)

# 2) matplotlib.rcParams[]으로 전역글꼴 설정 방법
# font_name = fm.FontProperties(fname=font_location).get_name()
# mpl.rc('font', family=font_name)

# 3) matplotlib.rcParams[]으로 전역글꼴 설정 방법
# plt.rcParams["font.family"] = 'Nanum Brush Script OTF'
# plt.rcParams["font.family"] = 'NanumGothic'

''' Test 그림그리기
import numpy as np
data = np.random.randint(-100, 100, 50).cumsum()
plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이')
plt.ylabel('주식 가격')
plt.xlabel('시간(분)')
plt.style.use('seaborn-pastel')
plt.show()
'''

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ml"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
def plot_feature_importances(model, data, feature_names):
    n_features = data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)


import graphviz 

def save_graph_as_svg(dot_string, output_file_name):
    if type(dot_string) is str:
        g = graphviz.Source(dot_string)
    elif isinstance(dot_string, (graphviz.dot.Digraph, graphviz.dot.Graph, graphviz.files.Source)):
        g = dot_string
    else:
        print('지원하지 않는 그림형식')
        return 

    g.format='svg'
    g.filename = output_file_name
    g.directory = './images/svg/'
    g.render(view=True)
    return g

'''   Test 그림그리기  
dot_graph = """
graph graphname {
    rankdir=LR;
     a -- b -- c;
     b -- d;
}"""
save_graph_as_svg(dot_graph, 'simple_dot_example2')
'''

