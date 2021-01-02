### make dataset
```
python dataset_tool.py create_from_images user/datasets user/images
```

### train
```
python train.py --outdir=user/snap --data=user/datasets --snap=4

### generate
python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

### generate with truncate
python generate.py --outdir=out --trunc=1.0 --seeds=600-605 --network=Cy8er-6MV.pkl

### generate with truncate and number for images to be generated 
python generate.py --outdir=out --trunc=1.0 --seeds=600-605 --network=Cy8er-6MV.pkl --sum=300


```