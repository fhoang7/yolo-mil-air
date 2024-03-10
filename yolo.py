#%% imports
from ultralytics import YOLO
import os
import torch
os.chdir("C:/Users/frank/Documents/GitHub/yolo-mil-air")
#%% Setup GPU

print(torch.__version__)
print(torch.cuda.is_available())
# %%
model = YOLO("yolov8n.pt")  
model.to('cuda')
#%%
results = model.train(data = 'data.yaml', epochs = 100)
# %%
