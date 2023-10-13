import torch
import torch.utils.data

class BaseThroughputDataset(torch.utils.data.Dataset):
    def __init__(self,
                 num_frames: int = 4,
                 use_audio: bool = True,
                 num_videos: int = 1000,
                 **kwargs,
                 ):

        self._num_videos = num_videos
        self.use_audio = use_audio

        # Load data in cpu beforehand
        self.dataset = []

        if kwargs['mats']:
            self._num_frames = int(num_frames * 1.5)
            print("check mats")
        else:
            self._num_frames = num_frames

        for i in range(self._num_videos):
            sample_dict = {
                "video_data": [torch.randn((self._num_frames,3,224,224))],
            }
            if self.use_audio:
                sample_dict["audio_data"] = [torch.randn((1,1024,128))]
            self.dataset.append(sample_dict)

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return self._num_videos

    def __getitem__(self, video_index):
        ret = self.dataset[video_index]
        return ret

    def __len__(self):
        return self._num_videos

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        video_keys = set([k for k in keys if "video" in k])
        audio_keys = set([k for k in keys if "audio" in k])
        other_keys = keys - video_keys - audio_keys

        # Change list formed data into tensor, extend batch size if more than one data in sample
        new_batch = []
        for sample in batch:
            while len(sample['video_data']) != 0:
                copied_dict = {k: sample[k] if k in other_keys else sample[k].pop() for k in keys}
                new_batch.append(copied_dict)

        batch_size = len(new_batch)
        dict_batch = {k: [dic[k] if k in dic else None for dic in new_batch] for k in keys}

        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]

        for size in video_sizes:
            assert (
                    len(size) == 4
            ), f"Collate error, an video should be in shape of (T, 3, H, W), instead of given {size}"

        if len(video_keys) != 0:
            max_video_length = self._num_frames
            max_height = max([i[2] for i in video_sizes])
            max_width = max([i[3] for i in video_sizes])

        for video_key in video_keys:
            video = dict_batch[video_key]
            new_videos = torch.ones(batch_size, max_video_length, 3, max_height, max_width) * -1.0
            for bi in range(batch_size):
                orig_batch = video[bi]
                if orig_batch is None:
                    new_videos[bi] = None
                else:
                    orig = video[bi]
                    new_videos[bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig
            dict_batch[video_key] = new_videos

        audio_sizes = list()
        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            for audio_i in audio:
                audio_sizes += [audio_i.shape]
        for size in audio_sizes:
            assert (
                    len(size) == 3
            ), f"Collate error, an audio should be in shape of (1, H, W), instead of given {size}"
        if len(audio_keys) != 0:
            max_height = max([i[1] for i in audio_sizes])
            max_width = max([i[2] for i in audio_sizes])

        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            new_audios = torch.ones(batch_size, 1, max_height, max_width) * -1.0
            for bi in range(batch_size):
                orig_batch = audio[bi]
                if orig_batch is None:
                    new_audios[bi] = None
                else:
                    orig = audio[bi]
                    new_audios[bi, : orig.shape[0], : orig.shape[1], : orig.shape[2]] = orig
            dict_batch[audio_key] = new_audios

        return dict_batch
