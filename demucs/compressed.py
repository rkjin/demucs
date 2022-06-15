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
            
        self.metadata.sort(key=lambda x: x["name"])
        self.duration = duration
        self.stride = stride
        self.channels = channels
        self.samplerate = samplerate
        self.streams = streams

    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata)

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
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            # print('getitem', index, examples, self.stride, self.duration, self.channels, self.samplerate, self.streams)
            #                   125      159         1        111647/8820         2             44100       slice(1, None, None)
            streams = AudioFile(meta["path"]).read(seek_time=index * self.stride,
                                                   duration=self.duration,
                                                   channels=self.channels,
                                                   samplerate=self.samplerate,
                                                   streams=self.streams) #torch.Size([4, 2, 558235])
                                  
            return (streams - meta["mean"]) / meta["std"]


def _get_track_metadata(path):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path)
    mix = audio.read(streams=0, channels=1, samplerate=44100)
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


def get_compressed_datasets(args, samples):
    metadata_file = args.metadata / "musdb.json" #metadata/musdb.json
    if not metadata_file.is_file() and args.rank == 0: # True
        _build_musdb_metadata(metadata_file, args.musdb, args.workers)
    if args.world_size > 1:
        distributed.barrier()
    metadata = json.load(open(metadata_file))
    duration = Fraction(samples, args.samplerate) # 558235 / 44100 -> 111647/8820
    stride = Fraction(args.data_stride, args.samplerate) # 44100 / 44100 -> 1
    
    train_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="train"),
                         metadata,
                         duration=duration,
                         stride=stride, #1
                         streams=slice(1, None),
                         samplerate=args.samplerate,
                         channels=args.audio_channels) # 111647/8820 1 slice(1, None, None) 44100 2
                  
    valid_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="valid"),
                         metadata,
                         samplerate=args.samplerate,
                         channels=args.audio_channels)
    return train_set, valid_set
