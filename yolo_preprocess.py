#%% Imports
import pandas as pd
import os
from pathlib import Path
import helpers
import shutil
from tqdm import tqdm
#%% Set source directory for original images/labels
os.chdir('C:/Users/frank/Documents/GitHub/yolo-mil-air/')
#%% Initialize directories
data_directory = Path(Path.cwd() / "datasets")
if not data_directory.exists():
    data_directory.mkdir()

labels_directory = Path(Path.cwd() / "datasets/labels")
if not labels_directory.exists():
    labels_directory.mkdir()

images_directory = Path(Path.cwd() / "datasets/images")
if not images_directory.exists():
    images_directory.mkdir()

#%% Define class id to object map
crop_files = sorted(os.listdir('../fighter-jets-photos/crop'))
class2idx = {class_name: i for i, class_name in enumerate(crop_files)}
idx2class = {i: class_name for i, class_name in enumerate(crop_files)}
#%% Figure out csvs and images
dataset_files = os.listdir('../fighter-jets-photos/dataset')
bbox_csvs = [f for f in dataset_files if f.endswith('.csv')]
images = [f for f in dataset_files if f.endswith('.jpg')]
#%% Convert CSVs into txt files
for csv in tqdm(bbox_csvs):
    df = pd.read_csv(f'../fighter-jets-photos/dataset/{csv}')
    yolo_df = helpers.convert_bboxes_to_yolo(df, class2idx)
    file_id = csv.split('.')[0]
    helpers.write_df_to_txt(yolo_df, labels_directory, file_id)
#%% Write images to correct file structure
os.chdir('C:/Users/frank/Documents/GitHub/fighter-jets-photos/dataset')
for pic in tqdm(images):
    shutil.copy(pic, images_directory)

# %%
