import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json

from tools import method_name_dict

palette_list=["#FF4E50", "#8C564B", "#5DADE2", "#A569BD", "#F5B041", "#1F77B4", "#594F4F"]
base_path = '/base_path'

def desaturate_color(color):
    hls_color = sns.color_palette("husl", n_colors=1, desat=0.5)[0]
    return sns.set_hls_values(color, s=hls_color[1])

palette_list=[desaturate_color(color) if idx!=len(palette_list)-1 else color for idx, color in enumerate(palette_list)]


def dot_plot(df, task, task_name, palettes):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create dot plot
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10, 6.5))
    ax = sns.lineplot(x="buffer_size", y=task_name, linewidth=4.0,
                      marker='o', markersize=16, data=df, hue="methods", palette=palettes)
    legend = ax.legend(loc='upper center', fontsize=20, ncol=3)
    # Set the line width in the legend
    for line in legend.get_lines():
        line.set_linewidth(3.5)

    # Set plot labels and title
    plt.title(task, fontsize=27, weight='bold')
    plt.xlabel('Rehearsal memory size', fontsize=25, weight='bold')
    plt.ylabel('Average accuracy', fontsize=25, weight='bold')

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.xticks(list(set(df["buffer_size"].to_list())), fontsize=25)
    plt.yticks(np.arange(int(minimum)-0.5, int(maximum)+1.0, 0.5), fontsize=25)

    plt.ylim([minimum-0.1, maximum+1.0])

    plt.show()


if __name__ == '__main__':

    targets = ['human','vehicle','nature','animal','others','home','music']
    methods = []
    seeds = []
    buffer_sizes = []

    palettes = palette_list[:len(methods)]

    classification_dir = os.path.join(base_path, 'FLAVA_code/experiments/checkpoints/cav_base_audioset_finetune_head/cav_base_audioset_finetune_head')
    summarize_df = pd.DataFrame()

    for buffer_size in buffer_sizes:

        acc_dict = {}
        for method in methods:
            acc_dict[method] = {}

            for target_id, target in enumerate(targets):
                acc_dict[method][target] = {}
                for random_seed in seeds:
                    acc_dict[method][target][random_seed] = []
                    for pretrain_dataset in targets[target_id:]:
                        if method == 'finetune':
                            path = os.path.join(
                                classification_dir + '_' + method + '_' + random_seed + '_' + pretrain_dataset + '_1', 'All',
                                'category_wise_top.json')
                        elif method == 'multitask':
                            path = os.path.join(
                                classification_dir + '_' + method + '_' + random_seed + '_' + 'All' + '_1', 'All',
                                'category_wise_top.json')
                        else:
                            path = os.path.join(classification_dir + '_' + method + '_' + random_seed + '_' + buffer_size + '_' + pretrain_dataset + '_1', 'All', 'category_wise_top.json')

                        if not os.path.isfile(path):
                            print("no", path, 'in the directory')
                            continue

                        with open(path, 'r') as f:
                            result = json.load(f)
                        top1_acc = result[target]['1'] * 100
                        acc_dict[method][target][random_seed].append(top1_acc)

        acc_avg_dict = {}
        num_tasks = len(targets)

        for method in methods:
            acc_avg_dict[method] = {}
            for seed in seeds:
                acc_avg_dict[method][seed] = 0

        for method in methods:
            for target_idx, target in enumerate(targets):
                for seed in seeds:
                    acc_list = np.array(acc_dict[method][target][seed])
                    if len(acc_list) == 0:
                        continue

                    last_task_acc = acc_list[-1]
                    acc_avg_dict[method][seed] += last_task_acc

        dfs = []
        for method in methods:
            new_df = {'methods': method_name_dict[method], 'buffer_size': buffer_size}
            for seed in seeds:
                acc_avg_dict[method][seed] /= num_tasks

            acc_avg_list = np.array(list(acc_avg_dict[method].values()))
            acc_avg_mean = np.round(np.mean(acc_avg_list), 2)
            acc_avg_std = np.round(np.std(acc_avg_list), 2)
            new_df[f"avg_acc"] = acc_avg_mean

            dfs.append(new_df)

        buffer_summary_df = pd.DataFrame(dfs)
        summarize_df = pd.concat([summarize_df, buffer_summary_df])

    summarize_df = summarize_df.reset_index(drop=True)
    dot_plot(summarize_df, 'Classification Acc', 'avg_acc', palettes)



