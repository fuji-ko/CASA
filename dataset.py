import os
import sys
import math
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from configs import TrainingConfig, DatasetConfig, MyAudioModelConfig
import h5py
from sklearn.utils import resample
from transformers import Wav2Vec2Config, WavLMConfig

class BaseDataset(Dataset):
    """Base dataset class with common functionality for all modalities"""
    
    labels = "SR,ISR,MUR,P,B,V,FG,HM,ME".split(",")
    primary_labels = "SR,ISR,MUR,P,B".split(",")
    secondary_labels = "V,FG,HM,ME".split(",")
    
    def __init__(self, 
                 root: str, 
                 annotator: str,
                 split: str = "train",
                 label: Optional[List[str]] = None):
        """
        Initialize base dataset
        
        Args:
            root: Root directory containing data
            label_path: Path to CSV file with labels
        """
        self.root = root
        self.annotator = annotator
        self.label = label 
        self.split = split

        self.data_df = self.prep_df()
        self.clips = self.data_df['clip_id']
         
    def prep_df(self):
        label_path = os.path.join(self.root, "labels", f"{self.annotator}.csv")
        df = pd.read_csv(label_path)
        df = df[df['split'] == self.split].reset_index(drop=True)

        if self.split == "test":
            return df
        
        if self.label:
    
            df_majority = df[df[self.label]==0]
            df_minority = df[df[self.label]==1]

            df_minority_upsampled = resample(df_minority,
                                                replace=True,    
                                                n_samples=len(df_majority),    
                                                random_state=42)
            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

            return df_upsampled
        else:
            return df

    def read_h5(self, h5_path: str, clip_id: str) -> Dict[str, torch.Tensor]:
        """Load all features for a clip_id from an HDF5 file."""
        features = {}
        with h5py.File(h5_path, 'r') as h5f:
            if clip_id not in h5f:
                raise KeyError(f"Clip {clip_id} not found in {h5_path}")
            group = h5f[clip_id]
            for k in group.keys():
                arr = group[k][()]
                features[k] = torch.tensor(arr, dtype=torch.float32)  
        
        return features
    
    def process_label(self, clip_data) -> Dict[str, int]:
        """Process labels from dataframe row"""
        labels: dict[str, int] = {}
        for label in self.labels:
            if label in clip_data and not pd.isna(clip_data[label]):
                labels[label] = clip_data[label] # clipdata[label] : int(0 or 1)
            else:
                labels[label] = None
        labels['any'] = any(val for val in labels.values() if val is not None)
        # labels['primary'] = any(val for val in labels.values() if val in self.primary_labels)
        # labels['secondary'] = any(val for val in labels.values() if val in self.secondary_labels)
        return labels # ex) labels={'SR': 0, 'ISR': 0, 'MUR': 0, 'P': 0, 'B': 0, 'V': 0, 'FG': 0, 'HM': 0, 'ME': 0, 'any': False}
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def get_item_paths(self, clip_id):
        """Get paths for a specific clip ID"""
        return {
            'audio': f"{self.root}/audios/{clip_id}.h5",
            'video': f"{self.root}/videos/{clip_id}.h5",
        }

class AudioDataset(BaseDataset):
    """Dataset for audio-only processing using Wav2Vec2"""
    
    def __init__(self, 
                 root: str, 
                 annotator: str, 
                 sampling_rate: int = 16000,
                 split: str = "train",
                 label: Optional[List[str]] = None,
                 **kwargs):
        
        self.target_sampling_rate = sampling_rate
        
        super().__init__(root, annotator, split, label)
    
    def __getitem__(self, idx):

        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        audio_path = self.get_item_paths(f"{task}_{media_file}")['audio']

        if os.path.exists(audio_path):
            audio = self.read_h5(audio_path, clip_id)
        else:
            print(f"Audio file not found for clip {clip_id}")
            audio = torch.zeros(1, self.target_sampling_rate)  # Default to empty audio

        label = self.process_label(clip_data)
        return {
            "clip_id": clip_id,
            "audio_inputs": audio,
            **label, # 辞書の中身を展開: {"A": ~, "B": ~} -> {"A": ~}, {"B": ~} 
        }

    
class FrameLevelAudioDataset(AudioDataset):
    def __init__(self, 
                 root, 
                 annotator, 
                 sampling_rate = 16000, 
                 split = "train", 
                 label = None, 
                 feature_extractor: str = "wav2vec2",
                 pretrained_model_name: str = 'facebook/wav2vec2-base-960h',
                 **kwargs):

        super().__init__(root, annotator, sampling_rate, split, label, **kwargs)
        self.feature_extractor = feature_extractor
        self.pretrained_model_name = pretrained_model_name
        self.target_sampling_rate = sampling_rate

    def __getitem__(self, idx):

        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        audio_path = self.get_item_paths(f"{task}_{media_file}")['audio']

        if os.path.exists(audio_path):
            audio = self.read_h5(audio_path, clip_id)
        else:
            print(f"Audio file not found for clip {clip_id}")
            audio = torch.zeros(1, self.target_sampling_rate)  # Default to empty audio

        label = self.process_annotationlabel(clip_data, self.feature_extractor, self.pretrained_model_name, self.target_sampling_rate) # 強ラベル作成

        return {
            "clip_id": clip_id,
            "audio_inputs": audio,
            **label, # 辞書の中身を展開: {"A": ~, "B": ~} -> {"A": ~}, {"B": ~} 
        }
    
    def process_annotationlabel(self, clip_data, feature_extractor, pretrained_model_name, sr) -> Dict[str, list]:
        """
        Process labels from dataframe row
        データフレームから強ラベルデータを作成
        特徴量抽出器の時刻フレームごとに吃音ありなしのラベルを作成
        kernel_size: 25ms(400samples(16kHz))
        stride_size: 20ms(320samples(16kHz))
        clip_dataは固定長を想定
        """
        # select feature extractor
        if feature_extractor == "wav2vec2":
            model_config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
        elif feature_extractor == "wavlm":
            model_config = WavLMConfig.from_pretrained(pretrained_model_name)
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        kernel_samples, stride_samples = self.calc_kernel_stride_samples(model_config.conv_kernel, model_config.conv_stride)

        labels: dict[str, list[int]] = {}
        annotation_start = clip_data["annotation_start"]
        annotation_end = clip_data["annotation_end"]
        start_time = clip_data["start_time"]
        end_time = clip_data["end_time"]
        clip_duration = (end_time - start_time) # float
        clip_samples = int(clip_duration * sr)
        num_frames = (clip_samples - kernel_samples) // stride_samples + 1 # 3 sec -> 149frames
        kernel_sec = kernel_samples / sr
        stride_sec = stride_samples / sr
        
        # annotationラベルが付与されてなければ，すべてのラベルの時刻フレームラベルを0にする
        if annotation_start is None or annotation_end is None:
            for label in self.labels:
                if label in clip_data and not pd.isna(clip_data[label]):
                    labels[label] = num_frames * [0] # num frame個分0が格納された0次元リスト
                else:
                    raise ValueError(f"予期しない条件分岐が行われました")

        # annotationラベルが付与されていれば，annotation区間に該当する時刻フレームに任意の症状ラベルをつける
        else:
            for label in self.labels:
                if label in clip_data and not pd.isna(clip_data[label]):
                    if clip_data[label] == 1:
                        labels[label] = self.make_stronglabel(num_frames, start_time, end_time, annotation_start,annotation_end, kernel_sec, stride_sec)
                    else: # if clip_data[label] == 0:
                        labels[label] = num_frames * [0]
                else:
                    raise ValueError(f"予期しない条件分岐が行われました")

        return labels
    
    def make_stronglabel(self, num_frames: int,
                         start_sec: float,
                         end_sec: float,
                         annotation_start: float,
                         annotation_end: float,
                         kernel_sec: float,
                         stride_sec: float) -> list[int]:
        """annotation区間に該当する時刻フレームに任意の症状ラベルをつける"""
        labels = [0] * num_frames
        
        for i in range(num_frames):
            frame_start = start_sec + i * stride_sec
            frame_end = frame_start + kernel_sec
            
            # フレームとアノテーション区間が少しでも重なっていれば1を返す
            if frame_start < annotation_end and frame_end > annotation_start:
                labels[i] = 1
            
        return labels

    def calc_kernel_stride_samples(self, kernels: list, strides: list):
        kernel_samples = 1
        stride_samples = 1

        for kernel, stride in zip(kernels, strides):
            kernel_samples += (kernel - 1) * stride_samples
            stride_samples *= stride

        return kernel_samples, stride_samples

        
class VideoDataset(BaseDataset):
    """Dataset for video-only processing using ViVIT"""
    
    def __init__(self, 
                 root: str, 
                 annotator: str,
                 split: str = "train",
                 **kwargs):
        
        super().__init__(root, annotator=annotator, split=split, label=None)

    def __getitem__(self, idx):
        clip_id = self.clips[idx]
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        video_path = self.get_item_paths(f"{task}_{media_file}")['video']

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for clip {clip_id}: {video_path}")
        
        video = self.read_h5(video_path, clip_id)
        label = self.process_label(clip_data)

        return {
            "clip_id": clip_id,
            "video_inputs": video,
            **label, # 辞書の中身を展開: {"A": ~, "B": ~} -> {"A": ~}, {"B": ~} 
        }
    
class VideoAudioDataset(BaseDataset):
    """Multimodal dataset for both video and audio processing"""
    
    def __init__(self, 
                 root: str, 
                 annotator: str, 
                 split: str = "train",
                 sampling_rate: int = 16000,
                 **kwargs):
        
        self.target_sampling_rate = sampling_rate
        super().__init__(root, annotator=annotator, split=split, label=None)

    def __getitem__(self, idx):
        clip_id = self.clips[idx]
        
        clip_data = self.data_df[self.data_df['clip_id'] == clip_id].iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        paths = self.get_item_paths(f"{task}_{media_file}")
        audio_path = paths['audio']
        video_path = paths['video']

        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            raise FileNotFoundError(f"file not found for clip {clip_id}: {audio_path}, {video_path}")
        
        audio_inputs = self.read_h5(audio_path, clip_id) 
        video_inputs = self.read_h5(video_path, clip_id)
        label = self.process_label(clip_data)
        return {
            "clip_id": clip_id,
            "audio_inputs": audio_inputs,
            "video_inputs": video_inputs,
            **label,
        }
    

def prep_dataset(config: DatasetConfig, 
                 split: str = "train", 
                 modality: Optional[str] = 'audio', 
                 feature_extractor: str = "wav2vec2", 
                 pretrained_model_name: str = 'facebook/wav2vec2-base-960h') -> Union[AudioDataset, VideoDataset, VideoAudioDataset]:

    if modality == "audio":
        return AudioDataset(**vars(config), split=split)
    elif modality == "exp_audio":
        return FrameLevelAudioDataset(**vars(config), split=split, feature_extractor=feature_extractor, pretrained_model_name=pretrained_model_name)
    elif modality == "video":
        return VideoDataset(**vars(config), split=split)
    elif modality == "multimodal":
        return VideoAudioDataset(**vars(config), split=split)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """dataloaderの返り値はこの関数の返り値とほぼ同等"""
    
    # collator to stack tensors
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], dict):
            collated[key] = collate_fn([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]

    return collated

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    config = DatasetConfig(
        root="data/clips",
        annotator="A1",
        sampling_rate=16000
    )
    
    dataset = prep_dataset(config, split="train", modality="multimodal")
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch['audio_inputs']['input_values'].shape, batch['video_inputs']['pixel_values'].shape)
        break  # Just print the first batch for testing