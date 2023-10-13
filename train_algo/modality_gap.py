#!/usr/bin/env python
import os
import json

import pandas as pd
import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm
from utils.distributed_ops import concat_all_gather
from utils.distributed_ops import all_gather
from collections import OrderedDict
from sklearn.decomposition import PCA
import pylab as plt
import seaborn as sns

def my_norm(x):
    return x/torch.norm(x, dim=-1, keepdim=True)

def modality_gap(loader, model, args, epoch=0, writer=None):

    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    category_dict = OrderedDict()
    for i, category in enumerate(args.data.target_task): # for i, category in enumerate(args.data.target_task):
        category_dict[category] = i
    print(category_dict)

    A_a_feat , A_v_feat, category_info = [], [], []
    with torch.no_grad():
        for sample in tqdm.tqdm(loader):
            batch_i = loader.batch_sampler.advance_batches_seen() - 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            keys = set([k for k in sample.keys() if "video" in k or "audio" in k or "label_idx" in k])
            input = {k: v.cuda(args.environment.gpu, non_blocking=True) for k,v in sample.items() if k in keys}
            input['masked_visual'] = False  # Do not mask when doing retrieval task
            input['masked_audio'] = False
            # input['modality_token'] = True
            input['retrieval'] = True

            with torch.cuda.amp.autocast():
                output = model(input)
                # mean pool, normalization all patches
                # audio_output = torch.mean(output['latent_c_a'], dim=1)
                # audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                # video_output = torch.mean(output['latent_c_v'], dim=1)
                # video_output = torch.nn.functional.normalize(video_output, dim=-1)
                audio_output = torch.mean(output['inter_c_a'], dim=1)
                video_output = torch.mean(output['inter_c_v'], dim=1)

            audio_output = audio_output.detach()
            video_output = video_output.detach()

            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

            category_indices = [category_dict[category] for category in sample['category']]
            category_indices = torch.tensor(category_indices, device=audio_output.device)
            category_info.append(category_indices)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    A_a_feat = torch.cat(A_a_feat, dim=0)
    A_v_feat = torch.cat(A_v_feat, dim=0)
    category_info = torch.cat(category_info, dim=0)

    A_a_feat = concat_all_gather(A_a_feat)
    A_v_feat = concat_all_gather(A_v_feat)
    category_info = concat_all_gather(category_info)
    mod_gap_dict = {}
    accum_mod_gap_dict = {}
    if writer is not None:

        # Get the gap
        for category_name, category_idx in category_dict.items():
            accum_cat_indices = torch.zeros(len(category_info)).bool()
            for prev_cat_idx in range(category_idx+1):
                category_indices = category_info == prev_cat_idx
                category_indices = category_indices.detach().cpu()
                accum_cat_indices = torch.logical_or(accum_cat_indices, category_indices)

            # modality gap until task category_idx
            accum_cat_mod_gap = torch.norm(my_norm(A_a_feat[accum_cat_indices]).mean(dim=0) -
                                            my_norm(A_v_feat[accum_cat_indices]).mean(dim=0))
            cat_mod_gap = torch.norm(my_norm(A_a_feat[category_indices]).mean(dim=0) -
                                             my_norm(A_v_feat[category_indices]).mean(dim=0))
            writer.add_scalar(f"{category_name}/accumulated_modality_gap", accum_cat_mod_gap, epoch)
            writer.add_scalar(f"{category_name}/modality_gap", cat_mod_gap, epoch)
            writer.flush()

            mod_gap_dict[category_name] = cat_mod_gap.detach().cpu().item()
            accum_mod_gap_dict[category_name] = accum_cat_mod_gap.detach().cpu().item()

        if os.path.isfile(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'modality_gap_result.json')):
            with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'modality_gap_result.json'), 'r') as f:
                task_wise_mod_gap_dict = json.load(f)
            task_wise_mod_gap_dict.update({epoch: mod_gap_dict})
        else:
            task_wise_mod_gap_dict = {}
            task_wise_mod_gap_dict[epoch] = mod_gap_dict
        os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix), exist_ok=True)
        with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'modality_gap_result.json'), 'w') as f:
            json.dump(task_wise_mod_gap_dict, f)

        if os.path.isfile(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'accumulated_modality_gap_result.json')):
            with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'accumulated_modality_gap_result.json'), 'r') as f:
                task_wise_accum_mod_gap_dict = json.load(f)
            task_wise_accum_mod_gap_dict.update({epoch: accum_mod_gap_dict})
        else:
            task_wise_accum_mod_gap_dict = {}
            task_wise_accum_mod_gap_dict[epoch] = accum_mod_gap_dict
        os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix), exist_ok=True)
        with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'accumulated_modality_gap_result.json'), 'w') as f:
            json.dump(task_wise_accum_mod_gap_dict, f)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()



def modality_gap_visualize(loader, model, args, epoch=0, writer=None):
    base_path = '/c1/jwlee'
    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    category_dict = OrderedDict()
    for i, category in enumerate(args.data.target_task):
        category_dict[category] = i

    A_a_feat , A_v_feat, category_info = [], [], []
    with torch.no_grad():
        for sample in tqdm.tqdm(loader):
            batch_i = loader.batch_sampler.advance_batches_seen() - 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            keys = set([k for k in sample.keys() if "video" in k or "audio" in k or "label_idx" in k])
            input = {k: v.cuda(args.environment.gpu, non_blocking=True) for k,v in sample.items() if k in keys}
            input['masked_visual'] = False  # Do not mask when doing retrieval task
            input['masked_audio'] = False
            # input['modality_token'] = True
            input['retrieval'] = True

            with torch.cuda.amp.autocast():
                output = model(input)
                # mean pool, normalization all patches
                # audio_output = torch.mean(output['latent_c_a'], dim=1)
                # audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                # video_output = torch.mean(output['latent_c_v'], dim=1)
                # video_output = torch.nn.functional.normalize(video_output, dim=-1)
                audio_output = torch.mean(output['inter_c_a'], dim=1)
                video_output = torch.mean(output['inter_c_v'], dim=1)

            audio_output = audio_output.detach()
            video_output = video_output.detach()

            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

            category_indices = [category_dict[category] for category in sample['category']]
            category_indices = torch.tensor(category_indices, device=audio_output.device)
            category_info.append(category_indices)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    A_a_feat = torch.cat(A_a_feat, dim=0)
    A_v_feat = torch.cat(A_v_feat, dim=0)
    category_info = torch.cat(category_info, dim=0)

    A_a_feat = concat_all_gather(A_a_feat)
    A_v_feat = concat_all_gather(A_v_feat)
    category_info = concat_all_gather(category_info)
    if writer is not None:

        # Get the gap
        for category_name, category_idx in category_dict.items():

            category_indices = category_info == category_idx
            category_indices = category_indices.detach().cpu()

            A_a_feat_np = np.array(my_norm(A_a_feat[category_indices]).detach().cpu())
            A_v_feat_np = np.array(my_norm(A_v_feat[category_indices]).detach().cpu())

            cat_mod_gap = np.linalg.norm(
                A_a_feat_np.mean(axis=0) - A_v_feat_np.mean(axis=0)
            )
            print(f"Category: {category_name} modality gap: {cat_mod_gap}")
            pca = PCA(n_components=6)
            pca.fit(np.concatenate((A_a_feat_np, A_v_feat_np), axis=0))

            pca_result_a = pca.transform(A_a_feat_np)
            df_a = pd.DataFrame()
            df_a['caption'] = ['Audio'] * len(A_a_feat_np)
            df_a['pca_one'] = pca_result_a[:,0]
            df_a['pca_two'] = pca_result_a[:,1]

            pca_result_v = pca.transform(A_v_feat_np)
            df_v = pd.DataFrame()
            df_v['caption'] = ['Video'] * len(A_v_feat_np)
            df_v['pca_one'] = pca_result_v[:,0]
            df_v['pca_two'] = pca_result_v[:,1]

            df_av = pd.concat([df_a, df_v], ignore_index=True)

            plt.figure(figsize=(10,7.3))
            ax = sns.scatterplot(
                x="pca_one", y="pca_two",
                hue='caption',
                data=df_av,
                s=50,
                palette=['#E74C3C','#1F77B4']
            )
            ax.legend(loc='upper left', fontsize=20.5)
            plt.title(category_name.capitalize() + " ("+args.logging.suffix+")", fontsize=27, weight='bold')
            plt.xlabel("")
            plt.ylabel("")
            if args.logging.suffix.startswith('AudioSet'):
                boundary=0.75
            else:
                boundary=0.75
            plt.xticks(np.arange(-boundary-0.1, boundary+0.3, 0.2), fontsize=20)
            plt.yticks(np.arange(-boundary-0.1, boundary+0.3, 0.2), fontsize=20)
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])
            for pair_id in range(len(A_a_feat_np)):
                x = [df_a['pca_one'][pair_id], df_v['pca_one'][pair_id]]
                y = [df_a['pca_two'][pair_id], df_v['pca_two'][pair_id]]
                plt.plot(x, y, '-', color='grey', alpha=0.1, linewidth=0.2)

            plt.text(x=0.34, y=0.1, s=f"Modality Gap: {round(cat_mod_gap.tolist(),3)}", transform=plt.gca().transAxes,
                     fontsize=20)

            # plt.show()
            plt.savefig(os.path.join(base_path, f'FLAVA_code/experiments/visualize/mod_vis_{args.logging.suffix}_{category_name}.pdf'), format='pdf', bbox_inches='tight')

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
