import pandas as pd
import shutil
#%% Convert CSVs into proper bounding box formats
def convert_bboxes_to_yolo(df: pd.DataFrame, class2idx: dict):
    """
    Returns a DF containing proper bboxes for YOLO
    """
    sample = df
    sample['class_id'] = sample['class'].apply(lambda x: class2idx[x]) 
    sample['xmin_norm'] = sample['xmin'] / sample['width']
    sample['xmax_norm'] = sample['xmax'] / sample['width']
    sample['ymin_norm'] = sample['ymin'] / sample['height']
    sample['ymax_norm'] = sample['ymax'] / sample['height']
    sample['xcenter'] = (sample['xmax_norm'] - sample['xmin_norm']) / 2
    sample['ycenter'] = (sample['ymax_norm'] - sample['ymin_norm']) / 2
    sample['width_norm'] = sample['xmax_norm'] - sample['xmin_norm']
    sample['height_norm'] = sample['ymax_norm'] - sample['ymin_norm']
    sample = sample[['class_id', 'xcenter', 'ycenter', 'width_norm', 'height_norm']]

    return sample

def write_df_to_txt(df: pd.DataFrame, txt_output_path, file_id: str):
    file_id_name = f'{file_id}.txt'
    text_file = txt_output_path / file_id_name
    if not text_file.exists():
        with text_file.open("w", encoding = "utf-8") as f:
            f.write(df.to_string(header = False, index = False))
    else:
        print(f'File {file_id_name} already exists.')
