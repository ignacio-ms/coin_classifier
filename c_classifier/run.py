from my_dataset import MyDataset

train = MyDataset()
train.read_data(datset_path='data/train/', verbose=True, shuffle=True)
train.print_img_by_index(0)

val = train.validation_split(0.8, verbose=True)
val.print_img_by_index(0)
