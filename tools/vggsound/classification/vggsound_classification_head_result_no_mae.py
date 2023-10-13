import pylab as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import json

from tools import method_name_dict

palette_list=["#594F4F", "#FF4E50", "#FF703F", "#5DADE2", "#A569BD", "#4CAF50", "#F5B041"]
base_path = '/base_path'

def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)


def bar_plot(df, metric, metric_name, palettes):

    # Set style and color palette
    sns.set(style="ticks", palette="Set1")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Create bar plot
    plt.figure()
    ax = sns.barplot(x="methods", y=metric, data=df, palette=palettes)

    # Set plot labels and title
    plt.title('Classification task', fontsize=24)
    plt.xlabel('Method', fontsize=24)
    plt.ylabel(metric_name, fontsize=24)

    maximum = df[metric].max()
    minimum = df[metric].min()

    plt.xticks(fontsize=20)
    plt.yticks(np.arange(int(minimum), int(maximum)+2, 2), fontsize=24)

    plt.ylim([minimum-1, maximum+1])

    change_width(ax, 0.25)

    plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
    plt.show()


if __name__=='__main__':

    targets = ['sports', 'music', 'vehicle', 'people', 'animals', 'home_nature', 'others_part1', 'others_part2']
    methods = []
    seeds = ['2021','2022','2023']
    buffer_sizes = ['2000']
    palettes = palette_list[:len(methods)]

    summarize_df = pd.DataFrame()
    methods_name = []
    for method in methods:
        methods_name.append(method_name_dict[method])
    summarize_df['methods'] = methods_name

    classification_dir = os.path.join(base_path, 'FLAVA_code/experiments/checkpoints/cav_base_vggsound_finetune_head/cav_base_vggsound_finetune_head')

    for buffer_size in buffer_sizes:

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

        forget_avg_dict = {}
        acc_avg_dict = {}
        num_tasks = len(targets)

        for method in methods:
            forget_avg_dict[method] = {}
            acc_avg_dict[method] = {}
            for seed in seeds:
                forget_avg_dict[method][seed] = 0
                acc_avg_dict[method][seed] = 0


        for method in methods:
            for target_idx, target in enumerate(targets):
                for seed in seeds:
                    acc_list = np.array(acc_dict[method][target][seed])
                    if len(acc_list) == 0:
                        continue

                    last_task_acc = acc_list[-1]
                    first_task_acc = acc_list[0]
                    acc_diff = acc_list[:-1] - last_task_acc
                    if len(acc_diff) != 0:
                        forget_avg_dict[method][seed] += np.max(acc_diff)
                    acc_avg_dict[method][seed] += last_task_acc


        forget_mean_dict = {}
        forget_std_dict = {}
        acc_mean_dict = {}
        acc_std_dict = {}

        for method in methods:
            for seed in seeds:
                if seed in forget_avg_dict[method]:
                        forget_avg_dict[method][seed] /= (num_tasks - 1)
                        acc_avg_dict[method][seed] /= num_tasks

            forget_avg_list = np.array(list(forget_avg_dict[method].values()))
            forget_mean_dict[method] = np.round(np.mean(forget_avg_list), 3)
            forget_std_dict[method] = np.round(np.std(forget_avg_list), 3)
            acc_avg_list = np.array(list(acc_avg_dict[method].values()))
            acc_mean_dict[method] = np.round(np.mean(acc_avg_list), 2)
            acc_std_dict[method] = np.round(np.std(acc_avg_list), 2)

            print(f"Environment:, {buffer_size}")

            print(method, f"avg_acc_mean: {acc_mean_dict[method]:.2f}, avg_acc_std: {acc_std_dict[method]:.2f}")
            print(method, f"forget_mean: {forget_mean_dict[method]:.2f}, forget_std: {forget_std_dict[method]:.2f}")

        summarize_df['avg_forgetting'] = list(forget_mean_dict.values())
        summarize_df['avg_acc'] = list(acc_mean_dict.values())
        summarize_df['fake_x'] = np.arange(0, len(methods))

        bar_plot(summarize_df, 'avg_forgetting', 'Average forgetting', palettes)
        bar_plot(summarize_df, 'avg_acc', 'Average accuracy', palettes)