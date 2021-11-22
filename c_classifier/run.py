from my_dataset import MyDataset
import data_preprocessing as eda


train = MyDataset()
train.read_data(datset_path='data/train/', verbose=True, shuffle=True)
