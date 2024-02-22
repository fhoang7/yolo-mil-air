import pandas as pd
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

    return df