"""
Reference:
    subplot:
        https://pyvisuall.tistory.com/68
        https://artiiicy.tistory.com/64
    Draw line between subplots:
        https://stackoverflow.com/questions/17543359/drawing-lines-between-two-plots-in-matplotlib
    Sub plot set label:
        https://stackoverflow.com/questions/6963035/how-to-set-common-axes-labels-for-subplots
    twinx to plot 2 graphs in 1 plot:
        https://rfriend.tistory.com/689
    subplot tick fontsize:
        https://stackoverflow.com/questions/38369188/set-size-of-ticks-in-all-subplots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from PIL import Image


def main(args):
    ##TODO: Input 3 images
    image1 = Image.open(f'./image_files/{args.image1}')
    image2 = Image.open(f'./image_files/{args.image2}')
    image3 = Image.open(f'./image_files/{args.image3}')

    fig = plt.figure()
    rows = 2; cols = 3

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.set_title('Epoch 0', fontsize=14)
    ax1.grid(False)
    ax1.set_xticks([])  ## To remove x grid
    ax1.set_yticks([])  ## To remove y grid
    ax1.imshow(image1)

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.set_title('Epoch 99', fontsize=14)
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(image2)

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.set_title('Epoch 249', fontsize=14)
    ax3.grid(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(image3)
    
    ##TODO: Convert .csv to line graph
    df_train_loss = pd.read_csv(f'./csv_files/{args.experiment}_train_loss.csv')
    ax4 = fig.add_subplot(212)
    ax4.set_xlabel('Epoch', fontsize=13)
    ax4.set_ylabel('Train Loss', fontsize=13)
    ax4.tick_params(axis='both', which='major', labelsize=13)
    ax4.plot(df_train_loss['Step'], df_train_loss['Train Loss'])

    df_rmse = pd.read_csv(f'./csv_files/{args.experiment}_rmse.csv')
    ax5 = ax4.twinx() 
    ax5.set_ylabel('RMSE', fontsize=13)
    ax5.tick_params(axis='both', which='major', labelsize=13)
    ax5.plot(df_rmse['Step'], df_rmse['Average RMSE'], color="black")
    
    ax4.legend(['Train Loss'], loc='best', fontsize=13)
    ax5.legend(['RMSE'], loc='upper center', fontsize=13)

    ##TODO: Draw vertical line
    fig.tight_layout()

    xy1 = (1, 1.24)
    xy2 = (255, 511)
    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data", axesA=ax4, axesB=ax1, color="red")
    ax4.add_artist(con)

    xy1 = (99, 1.05)
    xy2 = (255, 511)
    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data", axesA=ax4, axesB=ax2, color="red")
    ax4.add_artist(con)

    xy1 = (249, 0.98)
    xy2 = (255, 511)
    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data", axesA=ax4, axesB=ax3, color="red")
    ax4.add_artist(con)

    plt.savefig(f'{args.plot_name}') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image1', type=str, default="epoch0_val0_pred_gt.png")
    parser.add_argument('--image2', type=str, default="epoch99_val0_pred_gt.png")
    parser.add_argument('--image3', type=str, default="epoch249_val0_pred_gt.png")

    parser.add_argument('--experiment', type=str, default="MIDL_D60-5_W_P_progressive_weighted_erosion_ver2_1e-4_50")
    parser.add_argument('--plot_name', type=str, default="2023_MIDL_Loss_Graph")
    args = parser.parse_args()

    main(args)