# AKFEN semantic segmentation of HSI and RGB/PNG images using U-Net

## Dependencies

Before training the model necessary dependencies must be installed:

```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

chmod 750 Anaconda3-2022.05-Linux-x86_64.sh

./Anaconda3-2022.05-Linux-x86_64.sh

conda create -n  torch  python=3.9

conda activate torch

pip install torchsummary matplotlib

pip install Pillow

pip install spectral

pip install -U albumentations

pip install tqdm

apt-get install unzip
```

## For the RGB/PNG images

The PNG images of both Ripe and Unripe Quinces are used to train a model for the semantic segmentation using the U-Net architecture.

When training for the first time the variable `LOAD_MODEL` in the file `train.py` should be `False`.

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3.1 -c pytorch

git clone https://gitlab.edi.lv/astile.peter/akfen-semantic-segmentation.git

cd akfen-semantic-segmentation

cd png_unet

mkdir saved_images

python train.py
```
`LOAD_MODEL` can be changed to `True` if there is need for retraining the model. The results will be saved to the `saved_images` folder.

## For the HSI images

The HSI images used for the training was acquired by Specim IQ. Each image has 204 channels and a size of 512X512.

```
cd ..

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

cd HSI_unet

wget https://makonis.edi.lv/s/nMo6nzELY3JmKWM/download

apt-get install unzip

unzip download

rm -f download

cd HSI_Quince_set

unzip Data_HSI.zip

rm -f Data_HSI.zip

cd ..

```

Since the images acquired by the specim will contain one header file and dat file. It is better to convert them into single numpy array. Execute the command below to get respective numpy arrays.

```
python converter.py

```

After getting the respected numpy array, use the commands to copy the masks to the new dataset of numpy arrays.

```
cp -r HSI_Quince_set/Data_HSI/train_masks new_data/

cp -r HSI_Quince_set/Data_HSI/val_masks new_data/

rm -rf HSI_Quince_set

```

For training the model run the `train.py` file:

```
python train.py

```

For testing on the images use `test.py` file:

```
python test.py

```

## Acknowledgement

This work is supported by the Latvian Council of Science under the project No. lzp-2020/1-0353 “Smart non-invasive phenotyping of raspberries and Japanese quinces using machine learning and hyperspectral and 3D imaging” AKFEN.





