# A self-supervised single image dehazing


## Implementations

The below sections details what python requirements are required to set up for the project. 
dataset.

### Dependencies
- PyTorch, NumPy, OpenCV

### Dataset
We have worked on ImageNet dataset for the training procedure. For the testing procedure on synthetic noisy images, we have used BSD68, KODAK24 and Set14 datasets. For real noisy images, we have used the SIDD benchmark, CC and PolyU datasets.


##Dataloader
For loading the data, run the following: 
```
python dataloader.py 
```

### Train
We have provided the training code only for the synthetic noisy images. For real noisy images, we will provide the code later.  
```
python train_synthetic.py 
```

### Test
To get the results of the testing procedure, write the following on your command prompt and run. 

```
python test_synthetic.py"
```
