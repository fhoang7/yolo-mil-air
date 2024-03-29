#%% imports
from ultralytics import YOLO
import os
import torch
os.chdir("C:/Users/frank/Documents/GitHub/yolo-mil-air")
#%% Setup GPU and check
print(torch.__version__)
print(torch.cuda.is_available())
# %%
model = YOLO("yolov8m.pt")  
model.to('cuda')
#%%
results = model.train(data = 'data.yaml', epochs = 50, batch = -1)
# %%
metrics = model.val(data='data.yaml')