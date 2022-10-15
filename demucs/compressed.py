# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from fractions import Fraction
from concurrent import futures

import musdb
from torch import distributed

from .audio import AudioFile


def get_musdb_tracks(root, *args, **kwargs): # /home/bj/data/dnn/cfnet_venv/music_data/musdb18, "", ""
    mus = musdb.DB(root, *args, **kwargs)
    return {track.name: track.path for track in mus}


class StemsSet:
    def __init__(self, tracks, metadata, duration=None, stride=1,
                 samplerate=44100, channels=2, streams=slice(None)):
        self.metadata = []
        for name, path in tracks.items():
            meta = dict(metadata[name])
            meta["path"] = path
            meta["name"] = name
            # print('meta \n',meta)
            # [{'duration': 171.24, 'std': 0.15485107898712158, 'mean': 2.176033376599662e-05, 'path': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/A Classic Education - NightOwl.stem.mp4', 'name': 'A Classic Education - NightOwl'}]
            self.metadata.append(meta)
            if duration is not None and meta["duration"] < duration:
                raise ValueError(f"Track {name} duration is too small {meta['duration']}")
            #self.metadata [{'duration': 171.24, 'std': 0.15485107898712158, 'mean': 2.176033376599662e-05, 'path': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/A Classic Education - NightOwl.stem.mp4', 'name': 'A Classic Education - NightOwl'}, {'duration': 196.608, 'std': 0.14958864450454712, 'mean': 0.0003039466100744903, 'path': "/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - Devil's Words.stem.mp4", 'name': "Actions - Devil's Words"}, {'duration': 176.603, 'std': 0.13618549704551697, 'mean': -0.0005381310475058854, 'path': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - South Of The Water.stem.mp4', 'name': 'Actions - South Of The Water'}, {'duration': 189.07, 'std': 0.14917349815368652, 'mean': -0.000972075795289129, 'path': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Aimee Norwich - Child.stem.mp4', 'name': 'Aimee Norwich - Child'}]

        self.metadata.sort(key=lambda x: x["name"]) # list.sort 
        self.duration = duration # 35089/3150 , 두번째 None
        self.stride = stride # 1
        self.channels = channels # 2
        self.samplerate = samplerate # 44100
        self.streams = streams # slice(1, None, None)  , 두번째 slice(None, None, None) None 은 전체
    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata) #691

    def _examples_count(self, meta):
        if self.duration is None:
            return 1
        else:
            return int((meta["duration"] - self.duration) // self.stride + 1)

    def track_metadata(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)

            if index >= examples:
                index -= examples
                continue
            return meta

    def __getitem__(self, index):
        #index 545, 
        for meta in self.metadata:
        # {'duration': 171.24, 'std': 0.15485107898712158, 'mean': 2.176033376599662e-05, 'path': '/content/demucs/data/musdb18/train/A Classic Education - NightOwl.stem.mp4', 'name': 'A Classic Education - NightOwl'}
        # {'duration': 196.608, 'std': 0.14958864450454712, 'mean': 0.0003039466100744903, 'path': "/content/demucs/data/musdb18/train/Actions - Devil's Words.stem.mp4", 'name': "Actions - Devil's Words"}
        # {'duration': 176.603, 'std': 0.13618549704551697, 'mean': -0.0005381310475058854, 'path': '/content/demucs/data/musdb18/train/Actions - South Of The Water.stem.mp4', 'name': 'Actions - South Of The Water'}
        # {'duration': 189.07, 'std': 0.14917349815368652, 'mean': -0.000972075795289129, 'path': '/content/demucs/data/musdb18/train/Aimee Norwich - Child.stem.mp4', 'name': 'Aimee Norwich - Child'}
            examples = self._examples_count(meta) # 161, 186, 166, 178
            if index >= examples: # 545, 384, 198, 32
                index -= examples
                continue
            streams = AudioFile(meta["path"]).read(seek_time=index * self.stride, # 32 * 1
                                                   duration=self.duration, # 35089 / 3150
                                                   channels=self.channels, # 2
                                                   samplerate=self.samplerate, # 44100
                                                   streams=self.streams) # slice[1, None, none]
            # streams torch.Size([4, 2, 491246])                       
            return (streams - meta["mean"]) / meta["std"] #normalize


def _get_track_metadata(path):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path)
    mix = audio.read(streams=0, channels=1, samplerate=44100) #
    # torch.Size([1, 7552000])
    # torch.Size([1, 8671232])
    # torch.Size([1, 7204864])
    # torch.Size([1, 7788544])
    # torch.Size([1, 8338432])
    # torch.Size([1, 9256960])
    # torch.Size([1, 8835072])
    # torch.Size([1, 7911424])
    return {"duration": audio.duration, "std": mix.std().item(), "mean": mix.mean().item()}


def _build_metadata(tracks, workers=10):
    pendings = []
    # with futures.ProcessPoolExecutor(workers) as pool:
    #     for name, path in tracks.items():
    #         pendings.append((name, pool.submit(_get_track_metadata, path)))
    # return {name: p.result() for name, p in pendings}
    for name, path in tracks.items():
        pendings.append((name, _get_track_metadata(path)))
    print('process pool executor end')        
    return {name: p for name, p in pendings}

def _build_musdb_metadata(path, musdb, workers):
    tracks = get_musdb_tracks(musdb)
    metadata = _build_metadata(tracks, workers)
    path.parent.mkdir(exist_ok=True, parents=True)
    json.dump(metadata, open(path, "w"))


def get_compressed_datasets(args, samples): #491246
    metadata_file = args.metadata / "musdb.json" #metadata/musdb.json
    if not metadata_file.is_file() and args.rank == 0: # True
        _build_musdb_metadata(metadata_file, args.musdb, args.workers)
    if args.world_size > 1:
        distributed.barrier()
    metadata = json.load(open(metadata_file))
    duration = Fraction(samples, args.samplerate) # 491246 / 44100 
    stride = Fraction(args.data_stride, args.samplerate) # 44100 / 44100 -> 1
    
    train_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="train"),
                         metadata,
                         duration=duration,
                         stride=stride, 
                         streams=slice(1, None),
                         samplerate=args.samplerate,
                         channels=args.audio_channels) # 35089/3150  1 slice(1, None, None) 44100 2
                  
    valid_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="valid"),
                         metadata,
                         samplerate=args.samplerate,
                         channels=args.audio_channels) # slice(None, None, None) 
    return train_set, valid_set
