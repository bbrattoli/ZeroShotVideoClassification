# Introduction

Official code of the CVPR 2020 paper 
```Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications```
[Website](https://bbrattoli.github.io/ZeroShotVideoClassification/) [Arxiv](https://arxiv.org/abs/2003.01455)

<!---## Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications-->
## Summary

Learn a video representation that can generalize to unseen actions. 
Semantic information are used as supervision. In particular, the visual representation is mapped into the Word2Vec embedding, where words semantically related are closer to each other in an euclidean sense.

## Checkpoints
The trained models, used to produce the numbers in the paper, can be downloaded [here](https://drive.google.com/open?id=1OR89Z4GxxQOuOw376sc0Db9lBU0TxKWb).

# Install
## Requirements
Run `install.sh` to get the uncommon libraries (faiss, tensorboardx, joblib) and the latest version of pytorch compatible with cuda 9.2 installed in the docker.

## Retrieve external assets
### Get the word2vec model
```
sudo chmod + assets/download_word2vec.sh
./assets/download_word2vec.sh
```

### Get C3D pretrained model
```
wget http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle -P /workplace/
```

# Running
The script `run.sh` shows an example of parameters for starting the training of the e2e model.

## Training

### train on Kinetics, test on [UCF101, HMDB51]. End2End mode
```
python3 main.py --n_epochs 150 --bs 22 --lr 1e-3 --network r2plus1d_18 --dataset kinetics2both --save_path PATH_TO_RESULT_FOLDER --nopretrained
```

### train on Kinetics, test on [UCF101, HMDB51, ActivityNet]. End2End mode
```
python3 main.py --n_epochs 150 --bs 22 --lr 1e-3 --network r2plus1d_18 --dataset kinetics2others --save_path PATH_TO_RESULT_FOLDER --nopretrained
```

### train on Kinetics, test on [UCF101, HMDB51]. Baseline mode
```
python3 main.py --n_epochs 150 --bs 22 --lr 1e-3 --network r2plus1d_18 --dataset kinetics2both --save_path PATH_TO_RESULT_FOLDER --fixed  
```

### train on Kinetics, test on [UCF101, HMDB51, ActivityNet]. Baseline mode
```
python3 main.py --n_epochs 150 --bs 22 --lr 1e-3 --network r2plus1d_18 --dataset kinetics2others --save_path PATH_TO_RESULT_FOLDER --fixed  
```

### train on Kinetics, test on [UCF101, HMDB51]. End2End mode pretrained on SUN
```
python3 main.py --n_epochs 150 --bs 22 --lr 1e-3 --network r2plus1d_18 --dataset kinetics2both --save_path PATH_TO_RESULT_FOLDER --weights [path_to_SUN_pretraining]
```

### train on Kinetics, test on [UCF101, HMDB51, ActivityNet]. End2End mode pretrained on SUN
```
python3 main.py --n_epochs 150 --bs 22 --lr 1e-3 --network r2plus1d_18 --dataset kinetics2others --save_path PATH_TO_RESULT_FOLDER --weights [path_to_SUN_pretraining]
```

