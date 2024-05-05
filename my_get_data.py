import pandas as pd
import logging
logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

def get_data(file_path: str =None) -> pd.DataFrame:
    """
    read csv file from url or local path
    and return pandas.DataFrame
    """
    if not file_path:
        file_path = 'https://raw.githubusercontent.com/AshishJangra27/Machine-Learning-with-Python-GFG/main/Linear%20Regression/data_for_lr.csv'
        
    df = pd.read_csv(filepath_or_buffer=file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    logger.info('Loaded raw data shape:' + str(df.shape))
    return df


if __name__ == '__main__':
    
    df = get_data()
    print(df.head())
    
