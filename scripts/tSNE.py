import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.manifold import TSNE

# import internal libs
from data import DataUtils

# define the plot function
def plot_2D_scatter(save_path: str,
                    df: pd.DataFrame,
                    filename: str,
                    title: str) -> None:
    """plot 2d scatter from dataframe

    Args:
        save_path (str): the path to save fig
        df (pd.DataFrame): the data
        filename (str): the filename
        title (str): the title

    Return:
        None
    """
    assert os.path.exists(save_path), "path {} does not exist".format(save_path)
    assert len(df.columns) == 3, "the dataframe should have 3 columns"
    fig, ax = plt.subplots(figsize=(4, 3)) 
    ax = sns.scatterplot(data=df, hue=df.columns[-1], x=df.columns[0], y=df.columns[1])
    
    # remove the legend
    ax.get_legend().remove()

    fig.tight_layout()

    # save the fig
    path = os.path.join(save_path, "{}.pdf".format(filename))
    fig.savefig(path)
    path = os.path.join(save_path, "{}.png".format(filename))
    fig.savefig(path)
    plt.close()


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument("--load_path", default=None, type=str,
                        help='the path of loading the predicted labels.')
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()
    return args


def main():
    # get the args.
    args = add_args()

    # load the data
    features, labels = DataUtils.read()
    # load the pred_labels
    try:
        pred_labels = np.load(os.path.join(args.load_path, "pred_labels.npy"))
    except:
        raise ValueError(f"{args.load_path} is invalid to load pred_labels.npy.")
    
    # select only the top-10 labels
    selected_indices = labels < 10
    selected_features = features[selected_indices]
    selected_pred_labels = pred_labels[selected_indices]
    selected_labels = labels[selected_indices]
    
    # t-SNE
    tsne = TSNE(n_components=2)
    selected_features_tsne = tsne.fit_transform(selected_features)
    print(f"selected_features_tsne.shape: {selected_features_tsne.shape}")

    # plot the t-SNE
    for key, labels_ in {"gt": selected_labels, "pred": selected_pred_labels}.items():
        data_tsne = np.vstack((selected_features_tsne.T, labels_)).T
        tsne_df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2', 'class'])
        tsne_df['class'] = tsne_df['class'].astype(str)

        tsne_df.to_csv(os.path.join(args.load_path, f"tsne_df_{key}.csv"), index=False)

        # plot tsne
        plot_2D_scatter(save_path = args.load_path,
                        df = tsne_df,
                        filename = f"tsne_{key}",
                        title = "t-SNE on features")


if __name__ == "__main__":
    main()


# # now t-SNE
# tsne = TSNE(n_components=2)
# features_tsne = tsne.fit_transform(features_df)
# data_tsne = np.vstack((features_tsne.T, sampled_labels)).T
# tsne_df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2', 'class'])
# tsne_df['class'] = tsne_df['class'].astype(str)

# # make save_path
# save_path = os.path.join(os.path.dirname(MODEL_PATH), "TSNE", f"{os.path.basename(MODEL_PATH).split('.')[0]}")
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# # save the tsne_df
# tsne_df.to_csv(os.path.join(save_path, "tsne_df.csv"), index=False)

# # plot tsne
# plot_2D_scatter(save_path = save_path,
#                 df = tsne_df,
#                 filename = "tsne_features_testset",
#                 title = "t-SNE on features of testset")