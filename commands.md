### make dataset
```
python dataset_tool.py create_from_images user/datasets user/images
```

### train
```
python train.py --outdir=user/snap --data=user/datasets --snap=4
```