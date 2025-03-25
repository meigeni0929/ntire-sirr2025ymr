## Requirements
---

torch=2.0

CUDA=11.8

python=3.8

```plain
pip install pip install einops ema-pytorch fsspec fvcore huggingface-hub matplotlib 
numpy opencv-python omegaconf pytorch-msssim scikit-image scikit-learn scipy tensorboard 
tensorboardx wandb timm
```

```plain
cd DCNv4_op
python setup.py build
python setup.py install
```

## Training
Place the trained model under` ./pretrained`.

Pretrained file link:

Download link for the training data `Alldata`.：

```plain
python ./train_new.py --name rdnet --dataset 0.2 
--model cls_reg --batchSize=4
```

For the first stage, the training data is under `Alldata`, with `train1` used for the first stage training of rdnet and `train2` used for the second stage training of rdnet.

```plain
python train_more.py
```

`train1`data is used for the training phase of Dahaze.

## Testing
Place the test image set under `Alldata\test\blended0`.

<font style="color:rgb(35, 45, 54);background-color:rgb(247, 242, 235);">The trained weight file is located in </font>`[./pretrained](about:blank)`.

```plain
rdnetcon2_050_00019000.pt              rdnet trained model.
refine_007_00002730.pt                 rdnet trained model.
dahazebest.pth                         dahaze trained model.
```

  
Please take a close look at the code and avoid testing errors. `<path to the main checkpoint> `is the path to the trained weight file.

```plain
python pad1024.py
python3 test_sirs.py --icnn_path <path to the main checkpoint> --resume
python ww.py
python padhq.py
```

<font style="color:rgb(35, 45, 54);background-color:rgb(247, 242, 235);">When testing Dahaze, please modify the </font>`<font style="color:rgb(35, 45, 54);">folder2</font>`<font style="color:rgb(35, 45, 54);background-color:rgb(247, 242, 235);"> and </font>`<font style="color:rgb(35, 45, 54);">output_folder</font>`<font style="color:rgb(35, 45, 54);background-color:rgb(247, 242, 235);"> in </font>`<font style="color:rgb(35, 45, 54);">padhq.py</font>`<font style="color:rgb(35, 45, 54);background-color:rgb(247, 242, 235);"> to: </font>`'./Alldata/test/trans'`

```plain
python pad1024.py
python test.py
python padhq.py
```

## Out
Each time the output `out` will be overwritten, so please save it. Finally, compare the outputs of several results and select the best images.

