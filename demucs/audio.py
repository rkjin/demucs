# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import subprocess as sp
from pathlib import Path

import julius
import numpy as np
import torch

from .utils import temp_filenames


def _read_info(path):
    stdout_data = sp.check_output([
        'ffprobe', "-loglevel", "panic",
        str(path), '-print_format', 'json', '-show_format', '-show_streams'
    ])
#   print('start ########### \n',json.loads(stdout_data.decode('utf-8')))
#  b'{
#     "streams": [
#                 {     "index": 0,
#                       "codec_name": "aac",
#                       "codec_long_name": "AAC (Advanced Audio Coding)",
#                       "profile": "LC",
#                       "codec_type": "audio",
#                       "codec_time_base": "1/44100",
#                       "codec_tag_string": "mp4a",
#                       "codec_tag": "0x6134706d",
#                       "sample_fmt": "fltp",
#                       "sample_rate": "44100",
#                       "channels": 2,
#                       "channel_layout": "stereo",
#                       "bits_per_sample": 0,
#                       "r_frame_rate": "0/0",
#                       "avg_frame_rate": "0/0",
#                       "time_base": "1/44100",
#                       "start_pts": 0,
#                       "start_time": "0.000000",
#                       "duration_ts": 8834421,
#                       "duration": "200.327007",
#                       "bit_rate": "256018",
#                       "max_bit_rate": "267488",
#                       "nb_frames": "8630",
#                       "disposition": {
#                                       "default": 1,
#                                       "dub": 0,
#                                       "original": 0,
#                                       "comment": 0,
#                                       "lyrics": 0,
#                                       "karaoke": 0,
#                                       "forced": 0,
#                                       "hearing_impaired": 0,
#                                       "visual_impaired": 0,
#                                       "clean_effects": 0,
#                                       "attached_pic": 0,
#                                       "timed_thumbnails": 0
#                                      },
#     \n               "tags": {\n  
#                               "language": "und",
# \n                            "handler_name": "SoundHandler"
# \n                            }\n  
#                 },\n   
#                 {     "index": 1,
#                       "codec_name": "aac",
#                       "codec_long_name": "AAC (Advanced Audio Coding)",
#                       "profile": "LC",
#                       "codec_type": "audio",
#                       "codec_time_base": "1/44100",
#                       "codec_tag_string": "mp4a",
#                       "codec_tag": "0x6134706d",
#                       "sample_fmt": "fltp",
#                       "sample_rate": "44100",
#                       "channels": 2,
#                       "channel_layout": "stereo",
#                       "bits_per_sample": 0,
#                       "r_frame_rate": "0/0",
#                       "avg_frame_rate": "0/0",
#                       "time_base": "1/44100",
#                       "start_pts": 0,
#                       "start_time": "0.000000",
#                       "duration_ts": 8834421,
#                       "duration": "200.327007",
#                       "bit_rate": "256018",
#                       "max_bit_rate": "267488",
#                       "nb_frames": "8630",
#                       "disposition": {\n  
#                                       "default": 0,
#                                       "dub": 0,
#                                       "original": 0,
#                                       "comment": 0,
#                                       "lyrics": 0,
#                                       "karaoke": 0,
#                                       "forced": 0,
#                                       "hearing_impaired": 0,
#                                       "visual_impaired": 0,
#                                       "clean_effects": 0,
#                                       "attached_pic": 0,
#                                       "timed_thumbnails": 0
#                                      },\n     
#                       "tags": {              
#                                 "language": "und",
#                                 "handler_name": "SoundHandler"
#                               }   
#                  },\n      
#                  {\n  "index": 2,
#                       "codec_name": "aac",
#                       "codec_long_name": "AAC (Advanced Audio Coding)",
#                       "profile": "LC",
#                       "codec_type": "audio",
#                       "codec_time_base": "1/44100",
#                       "codec_tag_string": "mp4a",
#                       "codec_tag": "0x6134706d",
#                       "sample_fmt": "fltp",
#                       "sample_rate": "44100",
#                       "channels": 2,
#                       "channel_layout": "stereo",
#                       "bits_per_sample": 0,
#                       "r_frame_rate": "0/0",
#                       "avg_frame_rate": "0/0",
#                       "time_base": "1/44100",
#                       "start_pts": 0,
#                       "start_time": "0.000000",
#                       "duration_ts": 8834421,
#                       "duration": "200.327007",
#                       "bit_rate": "256018",
#                       "max_bit_rate": "267488",
#                       "nb_frames": "8630",
#                       "disposition": {\n   
#                                       "default": 0,
#                                       "dub": 0,
#                                       "original": 0,
#                                       "comment": 0,
#                                       "lyrics": 0,
#                                       "karaoke": 0,
#                                       "forced": 0,
#                                       "hearing_impaired": 0,
#                                       "visual_impaired": 0,
#                                       "clean_effects": 0,
#                                       "attached_pic": 0,
#                                       "timed_thumbnails": 0
#                                      },\n         
#                      "tags": {    
#                               "language": "und",
#                               "handler_name": "SoundHandler"
#                              }  
#                  },  
#                  {\n  "index": 3,
#                       "codec_name": "aac",
#                       "codec_long_name": "AAC (Advanced Audio Coding)",
#                       "profile": "LC",
#                       "codec_type": "audio",
#                       "codec_time_base": "1/44100",
#                       "codec_tag_string": "mp4a",
#                       "codec_tag": "0x6134706d",
#                       "sample_fmt": "fltp",
#                       "sample_rate": "44100",
#                       "channels": 2,
#                       "channel_layout": "stereo",
#                       "bits_per_sample": 0,
#                       "r_frame_rate": "0/0",
#                       "avg_frame_rate": "0/0",
#                       "time_base": "1/44100",
#                       "start_pts": 0,
#                       "start_time": "0.000000",
#                       "duration_ts": 8834421,
#                       "duration": "200.327007",
#                       "bit_rate": "256018",
#                       "max_bit_rate": "267488",
#                       "nb_frames": "8630",
#                       "disposition": {\n     
#                                       "default": 0,
#                                       "dub": 0,
#                                       "original": 0,
#                                       "comment": 0,
#                                       "lyrics": 0,
#                                       "karaoke": 0,
#                                       "forced": 0,
#                                       "hearing_impaired": 0,
#                                       "visual_impaired": 0,
#                                       "clean_effects": 0,
#                                       "attached_pic": 0,
#                                       "timed_thumbnails": 0
#                                      },\n    
#                       "tags": { 
#                                  "language": "und",
#                                  "handler_name": "SoundHandler"
#                               }   
#                  },   
#                  {\n  "index": 4,
#                       "codec_name": "aac",
#                       "codec_long_name": "AAC (Advanced Audio Coding)",
#                       "profile": "LC",
#                       "codec_type": "audio",
#                       "codec_time_base": "1/44100",
#                       "codec_tag_string": "mp4a",
#                       "codec_tag": "0x6134706d",
#                       "sample_fmt": "fltp",
#                       "sample_rate": "44100",
#                       "channels": 2,
#                       "channel_layout": "stereo",
#                       "bits_per_sample": 0,
#                       "r_frame_rate": "0/0",
#                       "avg_frame_rate": "0/0",
#                       "time_base": "1/44100",
#                       "start_pts": 0,
#                       "start_time": "0.000000",
#                       "duration_ts": 8834421,
#                       "duration": "200.327007",
#                       "bit_rate": "256018",
#                       "max_bit_rate": "267488",
#                       "nb_frames": "8630",
#                       "disposition": {                  
#                                        "default": 0,
#                                        "dub": 0,
#                                        "original": 0,
#                                        "comment": 0,
#                                        "lyrics": 0,
#                                        "karaoke": 0,
#                                        "forced": 0,
#                                        "hearing_impaired": 0,
#                                        "visual_impaired": 0,
#                                        "clean_effects": 0,
#                                        "attached_pic": 0,
#                                        "timed_thumbnails": 0
#                                      },                    
#                        "tags": {   "language": "und",
#                                    "handler_name": "SoundHandler"
#                                }          
#                   },       
#                   {\n "index": 5,
#                       "codec_name": "png",
#                       "codec_long_name": "PNG (Portable Network Graphics) image",
#                       "codec_type": "video",
#                       "codec_time_base": "0/1",
#                       "codec_tag_string": "[0][0][0][0]",
#                       "codec_tag": "0x0000",
#                       "width": 512,
#                       "height": 512,
#                       "coded_width": 512,
#                       "coded_height": 512,
#                       "has_b_frames": 0,
#                       "sample_aspect_ratio": "1:1",
#                       "display_aspect_ratio": "1:1",
#                       "pix_fmt": "rgba",
#                       "level": -99,
#                       "color_range": "pc",
#                       "refs": 1,
#                       "r_frame_rate": "90000/1",
#                       "avg_frame_rate": "0/0",
#                       "time_base": "1/90000",
#                       "start_pts": 0,
#                       "start_time": "0.000000",
#                       "duration_ts": 18029430,
#                       "duration": "200.327000",
#                       "disposition": {  "default": 0,
#                                         "dub": 0,
#                                         "original": 0,
#                                         "comment": 0,
#                                         "lyrics": 0,
#                                         "karaoke": 0,
#                                         "forced": 0,
#                                         "hearing_impaired": 0,
#                                         "visual_impaired": 0,
#                                         "clean_effects": 0,
#                                         "attached_pic": 1,
#                                         "timed_thumbnails": 0
#                                       }                
#                   }             
#                 ],             
#                 "format": {   "filename": "/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/Al James - Schoolboy Facination.stem.mp4",
#                               "nb_streams": 6, 
#                               "nb_programs": 0,
#                               "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
#                               "format_long_name": "QuickTime / MOV",
#                               "start_time": "0.000000",
#                               "duration": "200.327000",
#                               "size": "32262082",
#                               "bit_rate": "1288376",
#                               "probe_score": 100,
#                               "tags": {  "major_brand": "isom",
#                                          "minor_version": "1",
#                                          "compatible_brands": "isom",
#                                          "creation_time": "2017-12-16T16:25:40.000000Z"
#                                       }\n 
#                            }\
#    }    
    return json.loads(stdout_data.decode('utf-8'))

class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams) #[0, 1, 2, 3, 4]

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None,
             temp_folder=None):
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.
            temp_folder (str or Path or None): temporary folder to use for decoding.


        """
        # print(self )
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/A Classic Education - NightOwl.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - South Of The Water.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - Devil's Words.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/Al James - Schoolboy Facination.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/Angels In Amplifiers - I'm Alright.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/AM Contra - Heart Peripheral.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Aimee Norwich - Child.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # AudioFile(path=/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - One Minute Smile.stem.mp4, samplerate=44100, channels=2, streams=5) 
        # print(np.array(range(len(self)))) #[0 1 2 3 4]
        # print(streams) #0
        streams = np.array(range(len(self)))[streams]  
        # print(streams) #0
        single = not isinstance(streams, np.ndarray) # True
        if single: # True
            streams = [streams] # list[]

        if duration is None: # True
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))
        with temp_filenames(len(streams)) as filenames: #'/tmp/tmpmt997s2i'
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]
            # print(command)
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Aimee Norwich - Child.stem.mp4', '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmp0z3tq49w']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', "/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - Devil's Words.stem.mp4", '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpjcmozil8']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - One Minute Smile.stem.mp4', '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpoo837qy6']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/Al James - Schoolboy Facination.stem.mp4', '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpqun54d8t']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/A Classic Education - NightOwl.stem.mp4', '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpne2_2p_h']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', "/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/Angels In Amplifiers - I'm Alright.stem.mp4", '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpuklyp0of']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/test/AM Contra - Heart Peripheral.stem.mp4', '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpr22xumft']
            # ['ffmpeg', '-y', '-loglevel', 'panic', '-i', '/home/bj/data/dnn/cfnet_venv/music_data/musdb18/train/Actions - South Of The Water.stem.mp4', '-map', '0:0', '-threads', '1', '-f', 'f32le', '-ar', '44100', '/tmp/tmpynagx7zr']            
            sp.run(command, check=True)
            wavs = []
            for filename in filenames:
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels()).t() #([2, 7788544]) ....
                if channels is not None: # 1
                    wav = convert_audio_channels(wav, channels)
                if target_size is not None: #false
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    
    *shape, src_channels, length = wav.shape # ???????? *shape
    if src_channels == channels: # 2 == 1?
        pass
    elif channels == 1: # True
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels: #True
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels):
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)
