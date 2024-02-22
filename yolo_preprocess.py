#%% Imports
import pandas as pd
import os
#%% Read in a CSV
os.chdir('C:/Users/frank/OneDrive/Documents/Data Projects/yolo-mil-air/')
# sample_df = pd.read_csv('a csv.csv')
#%% Convert CSVs into proper bounding box formats
def convert_bboxes_to_yolo(df: pd.DataFrame, class2idx: dict):
    """
    Returns a DF containing proper bboxes for YOLO
    """
    pass
#%% Write annotations DF to txt

# Should take annotations 

#%% Create .txt files for each image containing bounding boxes