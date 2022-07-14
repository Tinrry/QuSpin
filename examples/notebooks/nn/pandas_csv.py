import pandas as pd


def add_dataframe(files):
    if len(files) == 0:
        print('no files')
        return None
    if len(files) == 1:
        return files[0]
    for i in range(len(files)):
        if i == 0:
            df = pd.read_csv(files[0], index_col=0)
        else:
            df2 = pd.read_csv(files[i], index_col=0)
            df = df.append(df2)
    combined_file = f'train_{df.shape[0]}_{df.shape[1] - 9}.csv'
    df.to_csv(combined_file)
    return combined_file


if __name__ == '__main__':
    train_files = ['train_2000_256.csv', 'train_3000_256.csv']
    combined_file = add_dataframe(train_files)
    print(pd.read_csv(combined_file).head(2).to_string())
