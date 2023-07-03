import os
import numpy as np
from glob import glob
from tqdm import tqdm
from spectral import *
from spectral.io import envi
from numpy import save

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)



def load_data(path):
	train_x = sorted(glob(os.path.join(path, "train_images", "*.hdr")))
	train_y = sorted(glob(os.path.join(path, "train_masks", "*.jpg")))
	
	val_x = sorted(glob(os.path.join(path, "val_images", "*.hdr")))
	val_y = sorted(glob(os.path.join(path, "val_masks", "*.jpg")))
	
	return (train_x, train_y), (val_x, val_y)
	
def create_data(images, masks, save_path):
	size = (512, 512)
	
	for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
		name = x.split("/")[-1].split(".")[0]		
		x = envi.open(x)
		x = np.array(x[:, :, :])
		save(f'{save_path+name}', x)
		
if __name__ == "__main__":
	
	
	""" Load the data """
	data_path = "HSI_Quince_set/Data_HSI/"
	load_data(data_path)
	
	(train_x, train_y), (val_x, val_y) = load_data(data_path)
	print(f"train : {len(train_x)} - {len(train_y)}")
	print(f"val : {len(val_x)} - {len(val_y)}")
	
	create_dir("new_data/train_images/")
	create_dir("new_data/val_images/")
	
	create_data(train_x, train_y, "new_data/train_images/")
	create_data(val_x, val_y, "new_data/val_images/")
