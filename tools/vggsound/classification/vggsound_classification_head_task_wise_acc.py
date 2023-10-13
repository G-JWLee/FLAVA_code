import pylab as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import json

from tools import method_name_dict

palette_list=["#FF703F", "#5DADE2", "#A569BD", "#4CAF50", "#F5B041"]
base_path = '/base_path'

def dot_plot(df, task, task_name, palettes):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create dot plot
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(10, 6.5))
    ax = sns.lineplot(x="task_id", y=task_name, marker='o', markersize=30, data=df, hue="methods", palette=palettes)

    # Set plot labels and title
    plt.title(task, fontsize=32)
    plt.xlabel('Pre-train task', fontsize=35)
    plt.ylabel('Average accuracy', fontsize=35)

    maximum = df[task_name].max()
    minimum = df[task_name].min()

    plt.xticks(list(set(df["task_id"].to_list())), fontsize=28)
    plt.yticks(np.arange(int(minimum)-1, int(maximum)+3, 2), fontsize=28)

    plt.ylim([minimum-1.0, maximum+3.0])

    # Increase legend box size
    ax.legend(loc='upper right', bbox_to_anchor=(1.00, 1.00), borderaxespad=0., fontsize=30)
    plt.show()

if __name__=='__main__':

    targets = ['sports', 'music', 'vehicle', 'people', 'animals', 'home_nature', 'others_part1', 'others_part2']
    methods = []
    seeds = []
    buffer_size = ''
    palettes = palette_list[:len(methods)]

    classification_dir = os.path.join(base_path, 'FLAVA_code/experiments/checkpoints/cav_base_vggsound_finetune_head/cav_base_vggsound_finetune_head')

    acc_dict = {}
    for method in methods:
        acc_dict[method] = {}

        for target_id, target in enumerate(targets):
            acc_dict[method][target] = {}
            for random_seed in seeds:
                acc_dict[method][target][random_seed] = []
                for pretrain_dataset in targets[target_id:]:

                    if method.startswith('finetune'):
                        path = os.path.join(
                            classification_dir + '_' + method + '_' + random_seed + '_' + pretrain_dataset + '_1', 'All',
                            'category_wise_top.json')
                    elif method == 'multitask' or method == 'no_pretrain_init':
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

    acc_list_dict = {}
    for method in methods:
        acc_list_dict[method] = {}

    for method in methods:
        for target_idx, target in enumerate(targets):

            acc_list = np.array(list(acc_dict[method][target].values()))
            acc_list = acc_list.mean(axis=0)
            acc_list_dict[method][target] = acc_list

    for target_idx, target in enumerate(targets):
        dfs = []
        for method in methods:
            for idx in range(len(acc_list_dict[method][target])):
                new_df = {'methods': method_name_dict[method]}
                new_df['acc'] = acc_list_dict[method][target][idx]
                new_df['task_id'] = (target_idx + 1) + idx

                dfs.append(new_df)

        target_summary_df = pd.DataFrame(dfs)
        dot_plot(target_summary_df, f"{target.capitalize()} Classification Acc", "acc", palettes)