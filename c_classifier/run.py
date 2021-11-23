from my_dataset import MyDataset
import data_preprocessing as eda


p_train = MyDataset(ds_type='np')
p_train.read_data(datset_path='data/train/')
p_train.save_data(new_father_path='p_data')
