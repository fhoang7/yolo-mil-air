#%% imports
from ultralytics import YOLO
import os
os.chdir("C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air")
# %%
model = YOLO("yolov8n.pt")  
#%%
results = model.train(data = 'data.yaml', epochs = 100)
# %%
