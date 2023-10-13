#!/usr/bin/env python
import os
import json
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

def get_sim_mat(a, b):
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    sim_mat = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mat.cpu().numpy()


def compute_metrics(x, category_mask=None):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d

    if category_mask is not None:
        ind = ind[category_mask]

    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

def print_computed_metrics_no_mr(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f}'.format(r1, r5, r10))

def retrieval(loader, model, args, epoch=0, writer=None):

    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    category_dict = {}
    for i, category in enumerate(loader.dataset.category_list): # for i, category in enumerate(args.data.target_task):
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
            input['modality_token'] = True
            input['retrieval'] = True

            with torch.cuda.amp.autocast():
                output = model(input)
                # mean pool, normalization all patches
                audio_output = torch.mean(output['latent_c_a'], dim=1)
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.mean(output['latent_c_v'], dim=1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)

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
    result_dict = {}
    if writer is not None:

        for direction in ['audio2video', 'video2audio']:
            if direction == 'audio2video':
                # audio -> video retrieval
                sim_mat = get_sim_mat(A_a_feat, A_v_feat)

            elif direction == 'video2audio':
                # video -> audio retrieval
                sim_mat = get_sim_mat(A_v_feat, A_a_feat)

            for category_name, category_idx in category_dict.items():
                if category_name not in result_dict.keys():
                    result_dict[category_name] = {}
                category_indices = category_info == category_idx
                category_indices = category_indices.detach().cpu()
                category_result = compute_metrics(sim_mat, category_indices)
                writer.add_scalar(f"{category_name}/{direction}_r1", category_result['R1'], epoch)
                writer.add_scalar(f"{category_name}/{direction}_r5", category_result['R5'], epoch)
                writer.add_scalar(f"{category_name}/{direction}_r10", category_result['R10'], epoch)
                writer.add_scalar(f"{category_name}/{direction}_MR", category_result['MR'], epoch)
                writer.flush()

                result_dict[category_name][direction] = category_result

            result = compute_metrics(sim_mat)
            print(direction)
            print_computed_metrics(result)
            writer.add_scalar(f"{direction}/r1", result['R1'], epoch)
            writer.add_scalar(f"{direction}/r5", result['R5'], epoch)
            writer.add_scalar(f"{direction}/r10", result['R10'], epoch)
            writer.add_scalar(f"{direction}/MR", result['MR'], epoch)
            writer.flush()
            result_dict[direction] = result

        if os.path.isfile(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json')):
            with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json'), 'r') as f:
                accum_result_dict = json.load(f)
            accum_result_dict.update({epoch: result_dict})
        else:
            accum_result_dict = {}
            accum_result_dict[epoch] = result_dict
        os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix), exist_ok=True)
        with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json'), 'w') as f:
            json.dump(accum_result_dict, f)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return result_dict



def tvlt_retrieval(loader_a, loader_v, model, args, epoch=0, writer=None):
    loader_v.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    category_dict = {}
    for i, category in enumerate(loader_v.dataset.category_list): # for i, category in enumerate(args.data.target_task):
        category_dict[category] = i

    audio_preload = list()
    aids = list()
    a_category_info = list()

    for _b in tqdm.tqdm(loader_a, desc="audio prefetch loop"):
        audio_preload.append(
            _b["audio_data"].cuda(args.environment.gpu, non_blocking=True),
        )
        aids += _b["index"]

        category_indices = [category_dict[category] for category in _b['category']]
        a_category_info += category_indices
    aids = torch.tensor(aids)
    a_category_info = torch.tensor(a_category_info)

    video_preload = list()
    for _b in tqdm.tqdm(loader_v, desc="video prefetch loop"):
        video_preload.append(
            [_b["video_data"].cuda(args.environment.gpu, non_blocking=True),
             _b["index"],
             [category_dict[category] for category in _b['category']]
             ],
        )

    rank_scores = list()
    rank_vids = list()
    rank_category_info = list()

    count = 0
    stop = 5
    with torch.no_grad():
        for video_batch in tqdm.tqdm(video_preload, desc="rank loop"):
            _ve, _vid, _v_category = video_batch
            b, l, c, h, w = _ve.shape
            _ve = _ve.unsqueeze(dim=1)

            video_batch_score = list()
            for audio_batch in audio_preload:
                fblen = len(audio_batch)
                ve = _ve.repeat(1, fblen, 1, 1, 1, 1)
                ve = ve.reshape(b*fblen, l, c, h, w)
                audio_batch = audio_batch.repeat(b,1,1,1)
                with torch.cuda.amp.autocast():
                    score = model.module.backbone.transformer.matching_score(
                        model.module.backbone.infer(
                            {
                                "audio_data": audio_batch,
                                "video_data": ve,
                            },
                            mask_audio=False,
                            mask_visual=False,
                            use_mae=False,
                            compute_joint_embedding=True,
                        )["cls_feats"]
                    )
                    score = F.sigmoid(score)
                    score = score.reshape(b,fblen)
                video_batch_score.append(score)

            video_batch_score = torch.cat(video_batch_score, dim=1)
            rank_scores.append(video_batch_score.cpu())
            rank_vids += _vid
            rank_category_info += _v_category

            # count+=1
            # if stop == count:
            #     break

    torch.distributed.barrier()
    rank_scores = torch.cat(rank_scores, dim=0)
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_vids = all_gather(rank_vids)
    gather_rank_category_info = all_gather(rank_category_info)

    vids = torch.tensor(gather_rank_vids)
    vids = vids.view(-1)
    category_info = torch.tensor(gather_rank_category_info)
    category_info = category_info.view(-1)
    scores = torch.cat(gather_rank_scores, dim=0)
    scores = scores.view(len(vids), -1)
    scores = scores.float() # topk operation does not support half precision

    result_dict = {}
    if writer is not None:

        for direction in ['audio2video', 'video2audio']:
            if direction == 'audio2video':
                # audio -> video retrieval
                topk10 = scores.topk(10, dim=0)
                topk5 = scores.topk(5, dim=0)
                topk1 = scores.topk(1, dim=0)
                topk10_vids = vids[topk10.indices]
                topk5_vids = vids[topk5.indices]
                topk1_vids = vids[topk1.indices]

                for category_name, category_idx in category_dict.items():
                    if category_name not in result_dict.keys():
                        result_dict[category_name] = {}
                    a_category_indices = a_category_info == category_idx
                    cat_topk10_vids = topk10_vids[:,a_category_indices]
                    cat_topk5_vids = topk5_vids[:,a_category_indices]
                    cat_topk1_vids = topk1_vids[:,a_category_indices]
                    cat_aids = aids[a_category_indices]

                    cat_a2v_r10 = (cat_aids.unsqueeze(0) == cat_topk10_vids).float().max(dim=0)[0].mean()
                    cat_a2v_r5 = (cat_aids.unsqueeze(0) == cat_topk5_vids).float().max(dim=0)[0].mean()
                    cat_a2v_r1 = (cat_aids.unsqueeze(0) == cat_topk1_vids).float().max(dim=0)[0].mean()

                    writer.add_scalar(f"{category_name}/{direction}_r10", cat_a2v_r10, epoch)
                    writer.add_scalar(f"{category_name}/{direction}_r5", cat_a2v_r5, epoch)
                    writer.add_scalar(f"{category_name}/{direction}_r1", cat_a2v_r1, epoch)
                    writer.flush()

                    result_dict[category_name][direction] = {
                        "R1": cat_a2v_r1.detach().cpu().item(), "R5": cat_a2v_r5.detach().cpu().item(), "R10": cat_a2v_r10.detach().cpu().item()
                    }

                a2v_r10 = (aids.unsqueeze(0) == topk10_vids).float().max(dim=0)[0].mean()
                a2v_r5 = (aids.unsqueeze(0) == topk5_vids).float().max(dim=0)[0].mean()
                a2v_r1 = (aids.unsqueeze(0) == topk1_vids).float().max(dim=0)[0].mean()

                result_dict[direction] = {
                    "R1": a2v_r1.detach().cpu().item(), "R5": a2v_r5.detach().cpu().item(), "R10": a2v_r10.detach().cpu().item()
                }
                print(direction)
                print_computed_metrics_no_mr(result_dict[direction])

                writer.add_scalar(f"{direction}/r10", a2v_r10, epoch)
                writer.add_scalar(f"{direction}/r5", a2v_r5, epoch)
                writer.add_scalar(f"{direction}/r1", a2v_r1, epoch)
                writer.flush()

            elif direction == 'video2audio':
                # audio -> video retrieval
                topk10 = scores.topk(10, dim=1)
                topk5 = scores.topk(5, dim=1)
                topk1 = scores.topk(1, dim=1)
                topk10_aids = aids[topk10.indices]
                topk5_aids = aids[topk5.indices]
                topk1_aids = aids[topk1.indices]

                for category_name, category_idx in category_dict.items():
                    if category_name not in result_dict.keys():
                        result_dict[category_name] = {}
                    v_category_indices = category_info == category_idx
                    cat_topk10_aids = topk10_aids[v_category_indices,:]
                    cat_topk5_aids = topk5_aids[v_category_indices,:]
                    cat_topk1_aids = topk1_aids[v_category_indices,:]
                    cat_vids = vids[v_category_indices]

                    cat_v2a_r10 = (cat_vids.unsqueeze(1) == cat_topk10_aids).float().max(dim=1)[0].mean()
                    cat_v2a_r5 = (cat_vids.unsqueeze(1) == cat_topk5_aids).float().max(dim=1)[0].mean()
                    cat_v2a_r1 = (cat_vids.unsqueeze(1) == cat_topk1_aids).float().max(dim=1)[0].mean()

                    writer.add_scalar(f"{category_name}/{direction}_r10", cat_v2a_r10, epoch)
                    writer.add_scalar(f"{category_name}/{direction}_r5", cat_v2a_r5, epoch)
                    writer.add_scalar(f"{category_name}/{direction}_r1", cat_v2a_r1, epoch)
                    writer.flush()

                    result_dict[category_name][direction] = {
                        "R1": cat_v2a_r1.detach().cpu().item(), "R5": cat_v2a_r5.detach().cpu().item(), "R10": cat_v2a_r10.detach().cpu().item()
                    }

                v2a_r10 = (vids.unsqueeze(1) == topk10_aids).float().max(dim=1)[0].mean()
                v2a_r5 = (vids.unsqueeze(1) == topk5_aids).float().max(dim=1)[0].mean()
                v2a_r1 = (vids.unsqueeze(1) == topk1_aids).float().max(dim=1)[0].mean()

                result_dict[direction] = {
                    "R1": v2a_r1.detach().cpu().item(), "R5": v2a_r5.detach().cpu().item(), "R10": v2a_r10.detach().cpu().item()
                }
                print(direction)
                print_computed_metrics_no_mr(result_dict[direction])

                writer.add_scalar(f"{direction}/r10", v2a_r10, epoch)
                writer.add_scalar(f"{direction}/r5", v2a_r5, epoch)
                writer.add_scalar(f"{direction}/r1", v2a_r1, epoch)
                writer.flush()

        if os.path.isfile(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json')):
            with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json'), 'r') as f:
                accum_result_dict = json.load(f)
            accum_result_dict.update({epoch: result_dict})
        else:
            accum_result_dict = {}
            accum_result_dict[epoch] = result_dict
        os.makedirs(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix), exist_ok=True)
        with open(os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix, 'retrieval_result.json'), 'w') as f:
            json.dump(accum_result_dict, f)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return result_dict
