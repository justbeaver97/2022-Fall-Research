"""
Reference:
    box plot:
        https://buillee.tistory.com/198
    box plot + scatter plot or jitter(swarm plot):
        https://danbi-ncsoft.github.io/study/2018/07/23/study_eda2.html
        https://gibles-deepmind.tistory.com/97
        https://blog.naver.com/youji4ever/221813848875
        https://buillee.tistory.com/198
    save seaborn figure:
        https://www.delftstack.com/ko/howto/seaborn/seaborn-save-figure/
    string.translate() to delete specific letter:
        https://www.freecodecamp.org/korean/news/paisseon-munjayeoleseo-munja-sagjehagi-how-to-delete-characters-from-strings/
    scatter plot:
        https://workingwithpython.com/matplotlib_scatterplot/
"""

import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def convert_txt_to_list(txt_file):
    rmse_list = [[0]*18 for _ in range(6)]

    with open(f"./txt_files/{txt_file}", "r") as f:
        for line in f:
            data1 = list(line.split(','))
    
    for i in range(len(data1)):
        rmse_value = data1[i].translate({ord('['): None})
        rmse_value = rmse_value.translate({ord(']'): None})
        rmse_list[i//18][i%18] = float(rmse_value)

    return rmse_list


def convert_list_to_dataframe(total_list, total_name):
    data = []
    for k in range(len(total_list)):
        for i in range(len(total_list[k])):
            for j in range(len(total_list[k][i])):
                data.append([i, total_list[k][i][j], total_name[k]])

    return pd.DataFrame(data, columns=["label", "value", "experiment"])


def box_plot(df, name):
    sns.boxplot(x=df['label'], y=df['value'], hue=df['experiment'], data=df)
    # sns.scatterplot(x=df['label'], y=df['value'], hue=df['experiment'], data=df)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Label', fontsize=13)
    plt.ylabel('RMSE (pixels)', fontsize=13)
    plt.legend(loc='upper center', fontsize=13)
    plt.savefig(f'{name}.png')
    plt.close()


def main(args):
    rmse_list_baseline1 = convert_txt_to_list(f"{args.baseline1_txt}")
    rmse_list_baseline2 = convert_txt_to_list(f"{args.baseline2_txt}")
    rmse_list_best      = convert_txt_to_list(f"{args.best_txt}")

    dataframe = convert_list_to_dataframe(
        [rmse_list_baseline1, rmse_list_baseline2, rmse_list_best],
        [args.experiment1, args.experiment2, args.experiment3]
    )
    box_plot(dataframe, f"{args.plot_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline1_txt', type=str, default="MSE_LIST_baseline_ConvNet_PE.txt")
    parser.add_argument('--baseline2_txt', type=str, default="MSE_LIST_baseline_ResNet.txt")
    parser.add_argument('--best_txt', type=str, default="MIDL_D60-10_W_P_progressive_weighted_erosion_ver2_1e-4_50_MSE_LIST.txt")

    parser.add_argument('--experiment1', type=str, default="ConvNet + PE")
    parser.add_argument('--experiment2', type=str, default="Pretrained ResNet")
    parser.add_argument('--experiment3', type=str, default="UNet + Dilation-Erosion")
    parser.add_argument('--plot_name', type=str, default="2023_MIDL_Box_Plot")
    args = parser.parse_args()

    main(args)