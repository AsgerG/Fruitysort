import os
import pandas as pd
import numpy as np
import json

data_path = "C:/Users/Asger/OneDrive/Dokumenter/Speciale/03_Fruit_sorter/data/"

def createDataframe(category_index, path_suffix,):

    # List files
    paths = os.listdir(data_path + "dataset/" + path_suffix)
    
    # Make labels
    labels = np.zeros(len(paths))
    labels = labels + category_index
    
    # Make full path
    new_paths = []
    for path in paths:
        new_paths.append(path_suffix + path)

    # Make DF
    d = {'paths': new_paths, 'label': labels}

    #os.chdir(current_path)
    
    df = pd.DataFrame(data=d)
    print(len(df))

    return df


def appendDatafremas(df_list):
    complete_df = df_list[0]
    for i, df in enumerate(df_list):
        if i != 0:
            complete_df = complete_df.append(df,ignore_index=True)
    print(len(complete_df))
    return complete_df





if __name__ == "__main__":

    # Create seperate Dataframes
    print("Number of items contained in each train Dataframe:")
    train_df_1 = createDataframe(0, "train/freshapples/")
    train_df_2 = createDataframe(1, "train/rottenapples/")
    train_df_3 = createDataframe(2, "train/freshbanana/")
    train_df_4 = createDataframe(3, "train/rottenbanana/")
    train_df_5 = createDataframe(4, "train/freshoranges/")
    train_df_6 = createDataframe(5, "train/rottenoranges/")
    print("\nNumber of items contained in each test Dataframe:")
    test_df_1 = createDataframe(0, "test/freshapples/")
    test_df_2 = createDataframe(1, "test/rottenapples/")
    test_df_3 = createDataframe(2, "test/freshbanana/")
    test_df_4 = createDataframe(3, "test/rottenbanana/")
    test_df_5 = createDataframe(4, "test/freshoranges/")
    test_df_6 = createDataframe(5, "test/rottenoranges/")


    # Append Dataframes
    print("\nNumber of items contained in complete train Dataframe:")    
    final_train_df = appendDatafremas((train_df_1, train_df_2, train_df_3, train_df_4, train_df_5, train_df_6))
    print("\nNumber of items contained in complete test Dataframe:")
    final_test_df = appendDatafremas((test_df_1, test_df_2, test_df_3, test_df_4, test_df_5, test_df_6))
    

    # Save as CSV
    final_train_df.to_csv(data_path + 'data_csv/train.csv', index=False)
    final_test_df.to_csv(data_path + 'data_csv/test.csv', index=False)


    # Save meta data 
    a_file = open(data_path + "meta_data/data.json", "w")
    data = {"train_load":len(final_train_df),
            "test_load":len(final_test_df)}   
    a_file = json.dump(data, a_file)