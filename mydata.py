'''
# 0.环境安装：下载 torchaudio -> pip install torchaudio
# import torchaudio
# from torch.utils.data import Dataset
# 1.下载数据集dataset_yesno
# torchaudio.datasets.YESNO()
# 2.定义自己的数据类
# __init__(self,dir,labe) -> None:
# __getitem__(self, index)
# __len__(self)
# 3.调用数据类
# mydata(dir,label)
'''

# 0.环境安装：下载
import os
import torchaudio
from torch.utils.data import Dataset
import plotf

# 1.下载数据集dataset_yesno
yes_no_dataset = torchaudio.datasets.YESNO(root = "./", download = True)
print("download successfully!")

# 2.定义自己的数据类
class Mydataset(Dataset):
    # __init__(self,dir,labe) -> None:
    def __init__(self, root_dir):
        self.path = root_dir
        self.audiofile = os.listdir(self.path)
    # __getitem__(self, index)
    def __getitem__(self, index):
        audio_name = self.audiofile[index]
        wave_path = os.path.join(self.path, audio_name)
        waveform, sr = torchaudio.load(wave_path)
        return waveform, sr, audio_name
    # __len__(self)
    def __len__(self):
        return len(self.audiofile)

# 3.调用数据类
# mydata(dir,label)
mydata = Mydataset(root_dir = "/Users/baijingyuan/jupyterPj/pytorch_practice/waves_yesno")
index = 3
waveform, sr, name = mydata[index]
plotf.plot_waveform(waveform.T[:, 0], sr)
print(name)