import os
import pathlib
import pandas as pd
def sum_up_result(result_folder, save_summary=False):

    D = dict()
    datasets = list()
    D["R_2"] = list()
    D["std_R_2"] = list()
    D["Q_2"] = list()
    D["std_Q_2"] = list()

    fit_time_sum = 0

    for folder in os.listdir(result_folder):
        if os.path.isdir(os.path.join(result_folder,folder)):
            if "CV_describe.csv" in os.listdir(os.path.join(result_folder, folder)):
                CV_result = os.path.join(result_folder, folder,"CV_describe.csv")
                df = pd.read_csv(CV_result, index_col="Unnamed: 0")
                R_2 = df.loc["mean"]["train_R2"]
                Q_2 = df.loc["mean"]["test_R2"]
                std_R_2 = df.loc["std"]["train_R2"]
                std_Q_2 = df.loc["std"]["test_R2"]
                mean_time = df.loc["mean"]["fit_time"]

                datasets.append(folder)
                D["R_2"].append(R_2)
                D["std_R_2"].append(std_R_2)
                D["Q_2"].append(Q_2)
                D["std_Q_2"].append(std_Q_2)

                fit_time_sum+= mean_time

    df_all = pd.DataFrame(D,index=datasets)
    print(df_all)
    if save_summary:
        print("\nSaving summary : ", os.path.join(result_folder,"summary.csv"))
        df_all.to_csv(os.path.join(result_folder,"summary.csv"))
    print("Total fit time : ", fit_time_sum)
    
    


if __name__ == "__main__":
    result_folder = "../results_linear/"
    # result_folder = "../results/"
    # result_folder = "../results_RNN/"

    save_summary = True
    sum_up_result(result_folder, save_summary)

