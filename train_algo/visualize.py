#!/usr/bin/env python
import os

import cv2
import einops
import tqdm
import numpy as np

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

from utils import distributed_ops
from utils.distributed_ops import concat_all_gather
import matplotlib
import matplotlib.pyplot as plt

import PIL.Image
import io
from torchvision.transforms import ToTensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models


def remove_axes():
    # Remove axes in plt.imshow()
    plt.axis('off')
    # Set transparent background
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # Set the background color to transparent
    plt.gca().set_facecolor((0, 0, 0, 0))
    # Display the plot without white borders
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), 100, 100, color='none', lw=0))


def make_audio_grid_tensor(audio):
    fig, axs = plt.subplots(1, 1)
    remove_axes()
    im = axs.imshow(audio[0].transpose(1, 0), origin="lower", aspect="auto")
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, transparent=True, format='jpeg')
    buf.seek(0)
    audio_grid = PIL.Image.open(buf)
    audio_grid = ToTensor()(audio_grid)

    return audio_grid



def attention_submodule_visualize(loader, model, args, category_name, writer=None):
    """
    Visualize attention map based on audio-video matching submodule.
    """
    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    # Compute embedding
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            batch_i = loader.batch_sampler.advance_batches_seen() - 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            keys = set([k for k in data.keys() if "video" in k or "audio" in k or "label_idx" in k])
            input_data = {k: v.cuda(args.environment.gpu, non_blocking=True) for k, v in data.items() if k in keys}
            input_data['return_attn'] = True
            pos_cross_attn_av, pos_cross_attn_va = model(input_data)
            pos_cross_attn_av = pos_cross_attn_av.detach().cpu()
            pos_cross_attn_va = pos_cross_attn_va.detach().cpu()

            input_data['video_data'] = input_data['video_data'].cpu()
            input_data['audio_data'] = input_data['audio_data'].cpu()

            # de-normalize frames
            mean_v = torch.as_tensor(IMAGENET_DEFAULT_MEAN)[None, None, :, None, None]
            std_v = torch.as_tensor(IMAGENET_DEFAULT_STD)[None, None, :, None, None]
            input_data['video_data'] = input_data['video_data'] * std_v + mean_v

            # de-normalize audio
            mean_a = torch.Tensor([-5.081])[None, :, None, None]
            std_a = torch.Tensor([4.4849])[None, :, None, None]
            input_data['audio_data'] = input_data['audio_data'] * std_a + mean_a

            for sample_idx in range(len(input_data['video_data'])):

                filename = data['vid'][sample_idx]

                # Visual heat map & masking visualization
                att_av = pos_cross_attn_av[sample_idx]
                images = input_data['video_data'][sample_idx]

                att_av = att_av / att_av.max()
                att_av = att_av.reshape(4, -1)  # T * (14*14)

                images_grid = torchvision.utils.make_grid(images)


                cams = []
                heat_maps = []

                n_images = (255 - (images * 255).long())  # T x C x H x W
                for image, map in zip(n_images, att_av):
                    width = int(map.size(0) ** 0.5)
                    map = map.reshape(width, width).detach().cpu().numpy()
                    map = cv2.resize(map, (image.shape[1], image.shape[2]))

                    heat_map = cv2.applyColorMap(np.uint8(255 * map), cv2.COLORMAP_JET)
                    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
                    heat_map = heat_map.transpose(2, 0, 1)
                    heat_map = np.float32(heat_map) / 255

                    image = np.float32(image) / 255

                    cam = heat_map + image
                    cam = cam / np.max(cam)
                    cams.append(np.uint8(255 * cam))
                    heat_maps.append(np.uint8(255 * heat_map))

                cams = np.stack(cams)
                cams = torch.from_numpy(cams)
                cams_grid = torchvision.utils.make_grid(cams)

                writer.add_image(f'{category_name}/{filename}_attention', cams_grid)
                writer.add_image(f'{category_name}/{filename}_images', images_grid)

                # Audio masking visualization
                spectrogram = input_data["audio_data"][sample_idx]

                att_va = pos_cross_attn_va[sample_idx]
                att_va = att_va.reshape(64, 8).numpy()

                # Original spectrogram
                spec = make_audio_grid_tensor(spectrogram)

                # Original heatmap
                att_va = att_va / np.max(att_va)
                att_va = cv2.resize(att_va, (spec.shape[-2], spec.shape[-1]))

                spec_heat_map = cv2.applyColorMap(np.uint8(255 * att_va), cv2.COLORMAP_JET)
                spec_heat_map = cv2.cvtColor(spec_heat_map, cv2.COLOR_BGR2RGB)
                spec_heat_map = spec_heat_map.transpose(2, 1, 0)
                spec_heat_map = torch.from_numpy(spec_heat_map)

                # Attention map
                spec_cam = np.float32(torch.flip(spec, [1])) + np.float32(spec_heat_map)/255
                spec_cam = spec_cam / np.max(spec_cam)
                spec_cam = np.uint8(255 * spec_cam)
                spec_cam = torch.from_numpy(spec_cam)
                spec_cam_grid = torchvision.utils.make_grid(spec_cam)
                spec_cam_grid = torch.flip(spec_cam_grid, [1])


                writer.add_image(f'{category_name}/{filename}_spec', spec)
                writer.add_image(f'{category_name}/{filename}_spec_attention', spec_cam_grid)

                writer.flush()

    print('finished!')



def attention_submodule_visualize_spurious(loader, model, args, category_name, writer=None):
    """
    Visualize attention map based on audio-video matching submodule.
    Together with current attention map, using the past data, we visualize attention map by past modality pair.
    """
    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    # Compute embedding
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            batch_i = loader.batch_sampler.advance_batches_seen() - 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            keys = set([k for k in data.keys() if "video" in k or "audio" in k or "label_idx" in k])
            input_data = {k: v.cuda(args.environment.gpu, non_blocking=True) for k, v in data.items() if k in keys}
            input_data['return_attn'] = True
            pos_cross_attn_av, pos_cross_attn_va, spu_cross_attn_av, spu_cross_attn_va = model(input_data)
            pos_cross_attn_av = pos_cross_attn_av.detach().cpu()
            pos_cross_attn_va = pos_cross_attn_va.detach().cpu()
            spu_cross_attn_av = spu_cross_attn_av.detach().cpu()
            spu_cross_attn_va = spu_cross_attn_va.detach().cpu()

            input_data['video_data'] = input_data['video_data'].cpu()
            input_data['audio_data'] = input_data['audio_data'].cpu()

            # de-normalize frames
            mean_v = torch.as_tensor(IMAGENET_DEFAULT_MEAN)[None, None, :, None, None]
            std_v = torch.as_tensor(IMAGENET_DEFAULT_STD)[None, None, :, None, None]
            input_data['video_data'] = input_data['video_data'] * std_v + mean_v

            # de-normalize audio
            mean_a = torch.Tensor([-5.081])[None, :, None, None]
            std_a = torch.Tensor([4.4849])[None, :, None, None]
            input_data['audio_data'] = input_data['audio_data'] * std_a + mean_a

            for sample_idx in range(len(input_data['video_data'])):

                filename = data['vid'][sample_idx]

                # Visual heat map & masking visualization
                att_av = pos_cross_attn_av[sample_idx]
                spu_att_av = spu_cross_attn_av[sample_idx]
                images = input_data['video_data'][sample_idx]

                original_v_max = np.maximum(att_av.max(), spu_att_av.max())

                att_av = att_av / original_v_max
                att_av = att_av.reshape(4, -1)  # T * (14*14)

                images_grid = torchvision.utils.make_grid(images)

                spu_att_av = spu_att_av / original_v_max
                spu_att_av = spu_att_av.reshape(4, -1)

                cams = []
                heat_maps = []

                spu_cams = []
                spu_heat_maps = []

                n_images = (255 - (images * 255).long())  # T x C x H x W
                for image, map, spu_map in zip(n_images, att_av, spu_att_av):
                    width = int(map.size(0) ** 0.5)
                    map = map.reshape(width, width).detach().cpu().numpy()
                    map = cv2.resize(map, (image.shape[1], image.shape[2]))
                    spu_map = spu_map.reshape(width, width).detach().cpu().numpy()
                    spu_map = cv2.resize(spu_map, (image.shape[1], image.shape[2]))

                    heat_map = cv2.applyColorMap(np.uint8(255 * map), cv2.COLORMAP_JET)
                    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
                    heat_map = heat_map.transpose(2, 0, 1)
                    heat_map = np.float32(heat_map) / 255
                    spu_heat_map = cv2.applyColorMap(np.uint8(255 * spu_map), cv2.COLORMAP_JET)
                    spu_heat_map = cv2.cvtColor(spu_heat_map, cv2.COLOR_BGR2RGB)
                    spu_heat_map = spu_heat_map.transpose(2, 0, 1)
                    spu_heat_map = np.float32(spu_heat_map) / 255

                    image = np.float32(image) / 255

                    cam = heat_map + image
                    cam = cam / np.max(cam)
                    cams.append(np.uint8(255 * cam))
                    heat_maps.append(np.uint8(255 * heat_map))

                    spu_cam = spu_heat_map + image
                    spu_cam = spu_cam / np.max(spu_cam)
                    spu_cams.append(np.uint8(255 * spu_cam))
                    spu_heat_maps.append(np.uint8(255 * spu_heat_map))

                cams = np.stack(cams)
                cams = torch.from_numpy(cams)
                cams_grid = torchvision.utils.make_grid(cams)

                spu_cams = np.stack(spu_cams)
                spu_cams = torch.from_numpy(spu_cams)
                spu_cams_grid = torchvision.utils.make_grid(spu_cams)

                writer.add_image(f'{category_name}/{filename}_attention', cams_grid)
                writer.add_image(f'{category_name}/{filename}_attention_spu', spu_cams_grid)
                writer.add_image(f'{category_name}/{filename}_images', images_grid)

                # Audio masking visualization
                spectrogram = input_data["audio_data"][sample_idx]

                att_va = pos_cross_attn_va[sample_idx]

                # Original spectrogram
                spec = make_audio_grid_tensor(spectrogram)

                spu_att_va = spu_cross_attn_va[sample_idx]
                att_va = att_va.reshape(64, 8).numpy()
                spu_att_va = spu_att_va.reshape(64,8).numpy()

                # Original heatmap
                original_a_max = np.maximum(np.max(att_va), np.max(spu_att_va))
                att_va = att_va / original_a_max
                att_va = cv2.resize(att_va, (spec.shape[-2], spec.shape[-1]))

                spec_heat_map = cv2.applyColorMap(np.uint8(255 * att_va), cv2.COLORMAP_JET)
                spec_heat_map = cv2.cvtColor(spec_heat_map, cv2.COLOR_BGR2RGB)
                spec_heat_map = spec_heat_map.transpose(2, 1, 0)
                spec_heat_map = torch.from_numpy(spec_heat_map)

                # Spurious heatmap
                spu_att_va = spu_att_va / original_a_max
                spu_att_va = cv2.resize(spu_att_va, (spec.shape[-2], spec.shape[-1]))

                spu_spec_heat_map = cv2.applyColorMap(np.uint8(255 * spu_att_va), cv2.COLORMAP_JET)
                spu_spec_heat_map = cv2.cvtColor(spu_spec_heat_map, cv2.COLOR_BGR2RGB)
                spu_spec_heat_map = spu_spec_heat_map.transpose(2, 1, 0)
                spu_spec_heat_map = torch.from_numpy(spu_spec_heat_map)

                # Attention map
                spec_cam = np.float32(torch.flip(spec, [1])) + np.float32(spec_heat_map)/255
                spec_cam = spec_cam / np.max(spec_cam)
                spec_cam = np.uint8(255 * spec_cam)
                spec_cam = torch.from_numpy(spec_cam)
                spec_cam_grid = torchvision.utils.make_grid(spec_cam)
                spec_cam_grid = torch.flip(spec_cam_grid, [1])

                spu_spec_cam = np.float32(torch.flip(spec, [1])) + np.float32(spu_spec_heat_map)/255
                spu_spec_cam = spu_spec_cam / np.max(spu_spec_cam)
                spu_spec_cam = np.uint8(255 * spu_spec_cam)
                spu_spec_cam = torch.from_numpy(spu_spec_cam)
                spu_spec_cam_grid = torchvision.utils.make_grid(spu_spec_cam)
                spu_spec_cam_grid = torch.flip(spu_spec_cam_grid, [1])

                writer.add_image(f'{category_name}/{filename}_spec', spec)
                writer.add_image(f'{category_name}/{filename}_spec_attention', spec_cam_grid)
                writer.add_image(f'{category_name}/{filename}_spec_attention_spu', spu_spec_cam_grid)

                writer.flush()

    print('finished!')



def attention_submodule_visualize_spurious_key_query_sim(loader, model, args, category_name, writer=None):
    """
    Visualize attention map based on audio-video matching submodule.
    Plus, we visualize the selected audio-video patches following our approach.
    """
    loader.batch_sampler.set_epoch(epoch=0)
    # switch to eval mode
    model.eval()

    # Compression ratio
    comp_v_ratio = 0.5
    comp_a_ratio = 0.5
    final_comp_v_ratio = 0.5
    final_comp_a_ratio = 0.5

    # Compute embedding
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            batch_i = loader.batch_sampler.advance_batches_seen() - 1
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            keys = set([k for k in data.keys() if "video" in k or "audio" in k or "label_idx" in k])
            input_data = {k: v.cuda(args.environment.gpu, non_blocking=True) for k, v in data.items() if k in keys}
            input_data['return_attn'] = True
            n_pos_cross_attn_av, att_av_indices, n_pos_cross_attn_va, att_va_indices, \
                cl_att_av_indices, cl_att_va_indices, v_prune_mat, a_prune_mat,\
                uni_pos_cross_attn_av, uni_spu_cross_attn_av, uni_pos_cross_attn_va, uni_spu_cross_attn_va = model(input_data)
            n_pos_cross_attn_av = n_pos_cross_attn_av.detach().cpu()
            n_pos_cross_attn_va = n_pos_cross_attn_va.detach().cpu()
            att_av_indices = att_av_indices.detach().cpu()
            att_va_indices = att_va_indices.detach().cpu()
            v_prune_mat = v_prune_mat.detach().cpu()
            a_prune_mat = a_prune_mat.detach().cpu()
            uni_pos_cross_attn_av = uni_pos_cross_attn_av.detach().cpu()
            uni_spu_cross_attn_av = uni_spu_cross_attn_av.detach().cpu()
            uni_pos_cross_attn_va = uni_pos_cross_attn_va.detach().cpu()
            uni_spu_cross_attn_va = uni_spu_cross_attn_va.detach().cpu()

            if cl_att_av_indices is not None:
                cl_att_av_indices = cl_att_av_indices.detach().cpu()
            if cl_att_va_indices is not None:
                cl_att_va_indices = cl_att_va_indices.detach().cpu()

            input_data['video_data'] = input_data['video_data'].cpu()
            input_data['audio_data'] = input_data['audio_data'].cpu()

            # de-normalize frames
            mean_v = torch.as_tensor(IMAGENET_DEFAULT_MEAN)[None, None, :, None, None]
            std_v = torch.as_tensor(IMAGENET_DEFAULT_STD)[None, None, :, None, None]
            input_data['video_data'] = input_data['video_data'] * std_v + mean_v

            # de-normalize audio
            mean_a = torch.Tensor([-5.081])[None, :, None, None]
            std_a = torch.Tensor([4.4849])[None, :, None, None]
            input_data['audio_data'] = input_data['audio_data'] * std_a + mean_a

            for sample_idx in range(len(input_data['video_data'])):

                filename = data['vid'][sample_idx]

                # Visual heat map & masking visualization
                att_av = n_pos_cross_attn_av[sample_idx]
                uni_att_av = uni_pos_cross_attn_av[sample_idx]
                uni_spu_att_av = uni_spu_cross_attn_av[sample_idx]
                att_av_ids = att_av_indices[sample_idx]
                num_compressed_tokens = int(len(att_av_ids) * (1-comp_v_ratio))
                images = input_data['video_data'][sample_idx]

                att_av_ids_keep = att_av_ids[:num_compressed_tokens]
                att_av_ids_restore = torch.argsort(att_av_ids)

                att_av = att_av / att_av.max()
                att_av = att_av.reshape(4, -1)  # T * (14*14)

                original_v_max = np.maximum(uni_att_av.max(), uni_spu_att_av.max())
                uni_att_av = uni_att_av / original_v_max
                uni_att_av = uni_att_av.reshape(4, -1)
                uni_spu_att_av = uni_spu_att_av / original_v_max
                uni_spu_att_av = uni_spu_att_av.reshape(4, -1)

                core_video_patches = einops.rearrange(images.transpose(0, 1), 'c t (h p0) (w p1) -> c (t h w) p0 p1',
                                                      p0=16, p1=16)
                core_video_patches = core_video_patches.transpose(0, 1)
                L = core_video_patches.shape[0]
                core_video_patches = torch.gather(core_video_patches, dim=0,
                                                  index=att_av_ids_keep[:, None, None, None].repeat(1, 3, 16, 16))
                zeros_video_data = torch.zeros(L - num_compressed_tokens, 3, 16, 16)
                core_video_patches = torch.cat([core_video_patches, zeros_video_data], dim=0)
                core_video_patches = torch.gather(core_video_patches, dim=0,
                                                  index=att_av_ids_restore[:, None, None, None].repeat(1, 3, 16, 16))
                core_video_patches = einops.rearrange(core_video_patches.transpose(0, 1),
                                                      'c (t h w) p0 p1 -> c t (h p0) (w p1)', t=4, h=14, w=14)
                core_video_patches = core_video_patches.transpose(0, 1)

                images_grid = torchvision.utils.make_grid(images)
                core_images_grid = torchvision.utils.make_grid(core_video_patches)

                cams = []
                heat_maps = []
                uni_cams = []
                uni_heat_maps = []
                uni_spu_cams = []
                uni_spu_heat_maps = []

                n_images = (255 - (images * 255).long())  # T x C x H x W
                for image, map, uni_map, uni_spu_map in zip(n_images, att_av, uni_att_av, uni_spu_att_av):
                    width = int(map.size(0) ** 0.5)
                    map = map.reshape(width, width).detach().cpu().numpy()
                    map = cv2.resize(map, (image.shape[1], image.shape[2]))
                    uni_map = uni_map.reshape(width, width).detach().cpu().numpy()
                    uni_map = cv2.resize(uni_map, (image.shape[1], image.shape[2]))
                    uni_spu_map = uni_spu_map.reshape(width, width).detach().cpu().numpy()
                    uni_spu_map = cv2.resize(uni_spu_map, (image.shape[1], image.shape[2]))

                    heat_map = cv2.applyColorMap(np.uint8(255 * map), cv2.COLORMAP_JET)
                    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
                    heat_map = heat_map.transpose(2, 0, 1)
                    heat_map = np.float32(heat_map) / 255

                    uni_heat_map = cv2.applyColorMap(np.uint8(255 * uni_map), cv2.COLORMAP_JET)
                    uni_heat_map = cv2.cvtColor(uni_heat_map, cv2.COLOR_BGR2RGB)
                    uni_heat_map = uni_heat_map.transpose(2, 0, 1)
                    uni_heat_map = np.float32(uni_heat_map) / 255

                    uni_spu_heat_map = cv2.applyColorMap(np.uint8(255 * uni_spu_map), cv2.COLORMAP_JET)
                    uni_spu_heat_map = cv2.cvtColor(uni_spu_heat_map, cv2.COLOR_BGR2RGB)
                    uni_spu_heat_map = uni_spu_heat_map.transpose(2, 0, 1)
                    uni_spu_heat_map = np.float32(uni_spu_heat_map) / 255

                    image = np.float32(image) / 255

                    cam = heat_map + image
                    cam = cam / np.max(cam)
                    cams.append(np.uint8(255 * cam))
                    heat_maps.append(np.uint8(255 * heat_map))

                    uni_cam = uni_heat_map + image
                    uni_cam = uni_cam / np.max(uni_cam)
                    uni_cams.append(np.uint8(255 * uni_cam))
                    uni_heat_maps.append(np.uint8(255 * uni_heat_map))

                    uni_spu_cam = uni_spu_heat_map + image
                    uni_spu_cam = uni_spu_cam / np.max(uni_spu_cam)
                    uni_spu_cams.append(np.uint8(255 * uni_spu_cam))
                    uni_spu_heat_maps.append(np.uint8(255 * uni_spu_heat_map))

                cams = np.stack(cams)
                cams = torch.from_numpy(cams)
                cams_grid = torchvision.utils.make_grid(cams)

                uni_cams = np.stack(uni_cams)
                uni_cams = torch.from_numpy(uni_cams)
                uni_cams_grid = torchvision.utils.make_grid(uni_cams)

                uni_spu_cams = np.stack(uni_spu_cams)
                uni_spu_cams = torch.from_numpy(uni_spu_cams)
                uni_spu_cams_grid = torchvision.utils.make_grid(uni_spu_cams)

                # If OCM, visualize the OCM-based masking also
                if cl_att_av_indices is not None:

                    att_av_ids = cl_att_av_indices[sample_idx]
                    num_compressed_tokens = int(len(att_av_ids) * (1-final_comp_v_ratio))
                    cl_att_av_ids_keep = att_av_ids[:num_compressed_tokens]
                    cl_att_av_ids_restore = torch.argsort(att_av_ids)

                    cl_core_video_patches = einops.rearrange(images.transpose(0, 1),
                                                          'c t (h p0) (w p1) -> c (t h w) p0 p1',
                                                          p0=16, p1=16)
                    cl_core_video_patches = cl_core_video_patches.transpose(0, 1)
                    L = cl_core_video_patches.shape[0]
                    cl_core_video_patches = torch.gather(cl_core_video_patches, dim=0,
                                                      index=cl_att_av_ids_keep[:, None, None, None].repeat(1, 3, 16, 16))
                    zeros_video_data = torch.zeros(L - num_compressed_tokens, 3, 16, 16)
                    cl_core_video_patches = torch.cat([cl_core_video_patches, zeros_video_data], dim=0)
                    cl_core_video_patches = torch.gather(cl_core_video_patches, dim=0,
                                                      index=cl_att_av_ids_restore[:, None, None, None].repeat(1, 3, 16,
                                                                                                           16))
                    cl_core_video_patches = einops.rearrange(cl_core_video_patches.transpose(0, 1),
                                                          'c (t h w) p0 p1 -> c t (h p0) (w p1)', t=4, h=14, w=14)
                    cl_core_video_patches = cl_core_video_patches.transpose(0, 1)
                    cl_core_images_grid = torchvision.utils.make_grid(cl_core_video_patches)

                prune_mask = v_prune_mat[sample_idx]
                prune_mask = prune_mask[None,:,None].repeat(1,1,768)
                unpatchify_video_prune_mask = models.objectives.unpatchify_video(prune_mask, 4, 3, 14, 14, 16)
                unpatchify_video_mask = unpatchify_video_prune_mask.squeeze(dim=0).bool()
                video_pruned = images.clone()
                video_pruned[unpatchify_video_mask] = 0
                video_pruned_grid = torchvision.utils.make_grid(video_pruned)

                writer.add_image(f'{category_name}/{filename}_attention', cams_grid)
                writer.add_image(f'{category_name}/{filename}_uni_attention', uni_cams_grid)
                writer.add_image(f'{category_name}/{filename}_uni_spu_attention', uni_spu_cams_grid)
                writer.add_image(f'{category_name}/{filename}_images', images_grid)
                writer.add_image(f'{category_name}/{filename}_core_images', core_images_grid)
                writer.add_image(f'{category_name}/{filename}_prune_images', video_pruned_grid)

                if cl_att_av_indices is not None:
                    writer.add_image(f'{category_name}/{filename}_cl_core_images', cl_core_images_grid)

                # Audio masking visualization
                spectrogram = input_data["audio_data"][sample_idx]

                att_va = n_pos_cross_attn_va[sample_idx]
                uni_att_va = uni_pos_cross_attn_va[sample_idx]
                uni_spu_att_va = uni_spu_cross_attn_va[sample_idx]

                att_va_ids = att_va_indices[sample_idx]
                att_va_ids_restore = torch.argsort(att_va_ids)
                audio_mask = torch.ones(len(att_va_ids))
                audio_mask[:int(len(audio_mask) * (1 - comp_a_ratio))] = 0
                audio_mask = torch.gather(audio_mask, dim=0, index=att_va_ids_restore)
                audio_mask = audio_mask[None,:, None].repeat(1, 1, 256)

                unpatchify_audio_mask = models.objectives.unpatchify(audio_mask, 1, 64, 8, 16)
                unpatchify_audio_mask = unpatchify_audio_mask.squeeze(dim=0).bool()
                audio_masked = spectrogram.clone().numpy()
                audio_masked[unpatchify_audio_mask] = np.nan

                current_cmap = matplotlib.cm.get_cmap()
                current_cmap.set_bad('black')

                audio_masked_spec = make_audio_grid_tensor(audio_masked)

                # Original spectrogram
                spec = make_audio_grid_tensor(spectrogram)

                att_va = att_va.reshape(64, 8).numpy()
                uni_att_va = uni_att_va.reshape(64, 8).numpy()
                uni_spu_att_va = uni_spu_att_va.reshape(64, 8).numpy()

                # Original heatmap
                att_va = att_va / np.max(att_va)
                att_va = cv2.resize(att_va, (spec.shape[-2], spec.shape[-1]))
                original_a_max = np.maximum(np.max(uni_att_va), np.max(uni_spu_att_va))
                uni_att_va = uni_att_va / original_a_max
                uni_att_va = cv2.resize(uni_att_va, (spec.shape[-2], spec.shape[-1]))
                uni_spu_att_va = uni_spu_att_va / original_a_max
                uni_spu_att_va = cv2.resize(uni_spu_att_va, (spec.shape[-2], spec.shape[-1]))

                spec_heat_map = cv2.applyColorMap(np.uint8(255 * att_va), cv2.COLORMAP_JET)
                spec_heat_map = cv2.cvtColor(spec_heat_map, cv2.COLOR_BGR2RGB)
                spec_heat_map = spec_heat_map.transpose(2, 1, 0)
                spec_heat_map = torch.from_numpy(spec_heat_map)

                uni_spec_heat_map = cv2.applyColorMap(np.uint8(255 * uni_att_va), cv2.COLORMAP_JET)
                uni_spec_heat_map = cv2.cvtColor(uni_spec_heat_map, cv2.COLOR_BGR2RGB)
                uni_spec_heat_map = uni_spec_heat_map.transpose(2, 1, 0)
                uni_spec_heat_map = torch.from_numpy(uni_spec_heat_map)

                uni_spu_spec_heat_map = cv2.applyColorMap(np.uint8(255 * uni_spu_att_va), cv2.COLORMAP_JET)
                uni_spu_spec_heat_map = cv2.cvtColor(uni_spu_spec_heat_map, cv2.COLOR_BGR2RGB)
                uni_spu_spec_heat_map = uni_spu_spec_heat_map.transpose(2, 1, 0)
                uni_spu_spec_heat_map = torch.from_numpy(uni_spu_spec_heat_map)

                # Attention map
                spec_cam = np.float32(torch.flip(spec, [1])) + np.float32(spec_heat_map)/255
                spec_cam = spec_cam / np.max(spec_cam)
                spec_cam = np.uint8(255 * spec_cam)
                spec_cam = torch.from_numpy(spec_cam)
                spec_cam_grid = torchvision.utils.make_grid(spec_cam)
                spec_cam_grid = torch.flip(spec_cam_grid, [1])

                uni_spec_cam = np.float32(torch.flip(spec, [1])) + np.float32(uni_spec_heat_map)/255
                uni_spec_cam = uni_spec_cam / np.max(uni_spec_cam)
                uni_spec_cam = np.uint8(255 * uni_spec_cam)
                uni_spec_cam = torch.from_numpy(uni_spec_cam)
                uni_spec_cam_grid = torchvision.utils.make_grid(uni_spec_cam)
                uni_spec_cam_grid = torch.flip(uni_spec_cam_grid, [1])

                uni_spu_spec_cam = np.float32(torch.flip(spec, [1])) + np.float32(uni_spu_spec_heat_map)/255
                uni_spu_spec_cam = uni_spu_spec_cam / np.max(uni_spu_spec_cam)
                uni_spu_spec_cam = np.uint8(255 * uni_spu_spec_cam)
                uni_spu_spec_cam = torch.from_numpy(uni_spu_spec_cam)
                uni_spu_spec_cam_grid = torchvision.utils.make_grid(uni_spu_spec_cam)
                uni_spu_spec_cam_grid = torch.flip(uni_spu_spec_cam_grid, [1])

                if cl_att_va_indices is not None:

                    att_va_ids = cl_att_va_indices[sample_idx]
                    num_compressed_tokens = int(len(att_va_ids) * (1-final_comp_a_ratio))
                    cl_att_va_ids_restore = torch.argsort(att_va_ids)

                    cl_mask = torch.ones(L)
                    cl_mask[:num_compressed_tokens] = 0
                    cl_mask = torch.gather(cl_mask, dim=0, index=cl_att_va_ids_restore)
                    cl_mask = cl_mask[None, :, None].repeat(1, 1, 256)

                    unpatchify_cl_mask = models.objectives.unpatchify(cl_mask, 1, 64, 8, 16)
                    unpatchify_cl_mask = unpatchify_cl_mask.squeeze(dim=0).bool()
                    audio_cl_masked = spectrogram.clone().numpy()
                    audio_cl_masked[unpatchify_cl_mask] = np.nan

                    current_cmap = matplotlib.cm.get_cmap()
                    current_cmap.set_bad('black')

                    cl_masked_spec = make_audio_grid_tensor(audio_cl_masked)

                prune_mask = a_prune_mat[sample_idx]
                prune_mask = prune_mask[None,:,None].repeat(1,1,256)
                unpatchify_audio_prune_mask = models.objectives.unpatchify(prune_mask, 1, 64, 8, 16)
                unpatchify_audio_mask = unpatchify_audio_prune_mask.squeeze(dim=0).bool()
                audio_pruned = spectrogram.clone().numpy()
                audio_pruned[unpatchify_audio_mask] = np.nan

                audio_pruned_grid = make_audio_grid_tensor(audio_pruned)

                writer.add_image(f'{category_name}/{filename}_spec', spec)
                writer.add_image(f'{category_name}/{filename}_core_spec', audio_masked_spec)
                writer.add_image(f'{category_name}/{filename}_prune_spec', audio_pruned_grid)

                writer.add_image(f'{category_name}/{filename}_spec_attention', spec_cam_grid)
                writer.add_image(f'{category_name}/{filename}_spec_uni_attention', uni_spec_cam_grid)
                writer.add_image(f'{category_name}/{filename}_spec_uni_spu_attention', uni_spu_spec_cam_grid)

                if cl_att_av_indices is not None:
                    writer.add_image(f'{category_name}/{filename}_cl_core_spec', cl_masked_spec)

                writer.flush()

    print('finished!')


def frames_visualize(loader, writer=None):
    """
    Visualize attention map based on audio-video matching submodule.
    """
    loader.batch_sampler.set_epoch(epoch=0)

    # Compute embedding
    with torch.no_grad():
        for data in tqdm.tqdm(loader):

            # de-normalize frames
            mean_v = torch.as_tensor(IMAGENET_DEFAULT_MEAN)[None, None, :, None, None]
            std_v = torch.as_tensor(IMAGENET_DEFAULT_STD)[None, None, :, None, None]
            data['video_data'] = data['video_data'] * std_v + mean_v

            # de-normalize audio
            mean_a = torch.Tensor([-5.081])[None, :, None, None]
            std_a = torch.Tensor([4.4849])[None, :, None, None]
            data['audio_data'] = data['audio_data'] * std_a + mean_a

            for sample_idx in range(len(data['video_data'])):

                filename = data['vid'][sample_idx]

                images = data['video_data'][sample_idx]
                images_grid = torchvision.utils.make_grid(images)

                # Audio masking visualization
                spectrogram = data["audio_data"][sample_idx]
                spec = make_audio_grid_tensor(spectrogram)

                writer.add_image(f'{filename}_images', images_grid)
                writer.add_image(f'{filename}_spec', spec)
                writer.flush()

    print('finished!')
