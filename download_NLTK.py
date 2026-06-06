# 首次运行需下载nltk的stemmer数据（仅需一次）
import nltk
# 使用清华 TUNA 镜像源
nltk.download('punkt', download_dir='./nltk_data', halt_on_error=False,
              download_url='https://pypi.tuna.tsinghua.edu.cn/simple')