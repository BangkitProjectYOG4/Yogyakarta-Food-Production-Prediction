import numpy as np
import pandas as pd

def preprocess(data, sliding_window=1):
    """
            Preprocessing the Data
            This function takes 2 arguments:
            1. data = the data that has been picked which
                        row we gonna use
                        type = pandas dataframe
            2. sliding_window = the value for number of features
                                that is used for prediction
                                type = integer
            Preprocessing is done thru below's steps:
            1. convert the type of data tuple into integer
            2. flatten the data since there is only one row
                from the original Dataset
            3. grouped the independent and it's dependent
                variable into 1 row
    """
    data.drop(data.columns[0], axis=True, inplace=True)
    data = (data
            .astype('int')
            .values
            .flatten()
            ).tolist()
    size = len(data) - sliding_window
    new = []
    for i in range(size):
        new.append(data[i:(i+1+sliding_window)])
    return (pd.DataFrame(new))


if __name__ == '__main__':
	path_to_file = 'data/padi.csv'
	province = 'DI YOGYAKARTA'

	df = pd.read_csv(path_to_file)
	df = df[df[df.columns[0]] == province]

	preprocessed = preprocess(df, 3)
	preprocessed.to_csv('data/preprocessed.csv')
	print('The preprocessed data has been saved as new csv file in data/preprocessed.csv')