#%% Imports
import pandas as pd
from helpers import convert_bboxes_to_yolo
#%% Set YOLO Data Directory
data_directory = "../yolo-jets"
#%% Read in a CSV
os.chdir('C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/')
#%% Figure class id to object map
crop_files = sorted(os.listdir('../fighter-jets-photos/crop'))
class2idx = {class_name: i for i, class_name in enumerate(crop_files)}
#%% Figure out csvs and images
dataset_files = os.listdir('../fighter-jets-photos/dataset')
bbox_csvs = [f for f in dataset_files if f.endswith('.csv')]
images = [f for f in dataset_files if f.endswith('.jpg')]
#%% Convert CSVs into txt files
sample = pd.read_csv(f'../fighter-jets-photos/dataset/{bbox_csvs[0]}')
#%% Write annotations DF to txt

# Should take annotations 

#%% Create .txt files for each image containing bounding boxes