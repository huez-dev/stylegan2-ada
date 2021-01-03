rem capture image from music video
python ImageGenerator4StyleGan/main.py

rem make traing data
python dataset_tool.py create_from_images user/datasets user/images
