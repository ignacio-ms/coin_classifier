from c_classifier.utils import timed
from c_classifier.nets.cnn import CNN
from c_classifier.data_preprocessing import noise_reduction, brightness_correction
from c_classifier.non_implemented.genetic_cnn import Genetic
from c_classifier.my_dataset import MyTfDataset, MyNpDataset

from nets.cnn import CNN
from nets.res_cnn import ResNet
from nets.tl_cnn import TransferVGG
