import os
import sys
import math
import soundfile as sf
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from configs import TrainingConfig, DatasetConfig
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
        
class Sep28kAudioDataset(Dataset):
    def __init__(self,
                 root,
                 split,
                 label,
                 sampling_rate,
                 classification_task,
                 **kwargs,
                 ):
        # sep-28kdatasetでは，開発データ群は"dev"で表される.
        if split == "val":
            split = "dev"

        self.root = root
        self.split = split
        self.label = label
        self.sampling_rate = sampling_rate
        self.label_path = "/work/abelab5/k_fuji/ml-stuttering-events-dataset-extended/SEP-28k-Extended_clips.csv"
        self.classification_task = classification_task
        self.symptom_names = ["SoundRep", "WordRep", "Prolongation", "Block", "Interjection"]
        
        self.data_df = self.prep_df()
        self.show = self.data_df["Show"]
        self.epid = self.data_df["EpId"]
        self.clipid = self.data_df["ClipId"]
        self.labels = "Prolongation,Block,SoundRep,WordRep".split(",")
    
    def prep_df(self):
        """正解クリップの前処理"""
        label_path = self.label_path
        df = pd.read_csv(label_path)
        print(f"all clips: {len(df)} clips")
        
        # trainとtestをまとめて学習に用いる
        selected_splits = (["train", "test"] if self.split == "train" else [self.split])
        
        # 所望のsplitのデータを取り出し，不要なデータは除外
        df = df[
            (df["SEP28k-E"].isin(selected_splits))
            & (df["NoSpeech"] < 2)
            & (df["Music"] < 2)
            & (df["DifficultToUnderstand"] < 2)
            & (df["PoorAudioQuality"] < 2)
            & (df["Unsure"] < 2)
            ].copy()
        
        # 5症状およびNoStutteredWordsが1以下のクリップは曖昧であるため除去
        has_symptom = df[self.symptom_names].ge(2).any(axis=1)
        has_no_stutter = df["NoStutteredWords"].ge(2)
        # 症状または無吃音の，少なくとも一方に2票以上あれば採用
        df = df[has_symptom | has_no_stutter].reset_index(drop=True)

        return df

    def read_wav(self, wav_path: str) -> Dict[str, torch.Tensor]:
        """Load all features for a clip_id from wav_dir"""
        features = {}
        wav, sr = sf.read(wav_path, dtype="float32")
        
        if wav.ndim != 1:
            raise ValueError("モノラル音声を指定してください")

        if sr != self.sampling_rate:
            raise ValueError("sampling_rateが異なります")
        features["input_values"] = torch.from_numpy(wav).unsqueeze(0) #要検討
        return features

    def process_label(self, clip_data) -> Dict[str, int]:
        """process labels from dataframe row"""
        active_symptoms = [
            name for name in self.symptom_names 
            if name in clip_data and not pd.isna(clip_data[name]) and clip_data[name] >= 2
        ]
        
        has_symptom = bool(active_symptoms)
        no_stutter_votes = clip_data["NoStutteredWords"]
        has_no_stutter = (no_stutter_votes) >= 2
        
        if not has_symptom and not has_no_stutter:
            raise ValueError("症状，無症状ともに2票未満の曖昧なクリップです")
        
        
        if self.classification_task == "multilabel":
            labels = {name: int(name in active_symptoms) for name in self.symptom_names}
            labels["any"] = int(has_symptom)
        
        elif self.classification_task == "multiclass":
            labels = {
                name: 0 for name in [*self.symptom_names, "no-stutter", "other"] 
            }
            
            if len(active_symptoms) >= 2:
                target = "other"
            elif len(active_symptoms) == 1:
                target = active_symptoms[0] # 先頭を取り出す(一つしか入ってないから)
            else:
                target = "no-stutter"
            
            labels[target] = 1
            
        elif self.classification_task == "binaryclass":
            labels = {
                "any": int(has_symptom),
                "no-stutter": int(not has_symptom)
            }
        
        else:
            raise ValueError("定義されていないclassification taskです")
        
        return labels

    def __len__(self) -> int:
        return len(self.data_df)

    def get_item_paths(self, show, epid, clipid):
        """Get paths for show, epid, clipid"""
        """sep-28kは二つのディレクトリにまたがって入っているので二つのパスを返す"""
        return {
            "sep-28k": f"/work/abelab5/k_fuji/sep-28k/clips/{show}/{epid}/{show}_{epid}_{clipid}.wav",
            "sep-28k-E": f"/work/abelab5/k_fuji/sep-28k-E/data/{show}/{show}_{epid}_{clipid}.wav"
        }

    def extract_audio_features(self, sep_28k_path, sep_28k_E_path):
        """sep_28k_pathとsep-28k-Eの音声を検索"""
        """1: 音声ファイルが存在するかを確認"""
        """2: 存在する場合，0サンプルでない方のpathから音声を抽出する"""
        """音声ファイルが存在しない場合や，0サンプルの音声ファイルしか存在しない場合，0埋めされたtensorを返す"""
        for path in (sep_28k_path, sep_28k_E_path):
                if not os.path.isfile(path):
                    continue

                audio = self.read_wav(path)
                values = audio["input_values"]  # (1, サンプル数)

                if values.shape[-1] == 3 * self.sampling_rate: # 3秒音声になっているならそのまま返す
                    return audio

        # どちらも存在しない、またはサンプル数が3秒未満であれば3秒のゼロ音声
        return {"input_values": torch.zeros(1, 3 * self.sampling_rate, dtype=torch.float32)}

    def __getitem__(self, idx):
        clip_data = self.data_df.iloc[idx]
        
        show = clip_data["Show"]
        epid = int(clip_data["EpId"])
        clipid = int(clip_data["ClipId"])

        paths = self.get_item_paths(show, epid, clipid)
        
        audio = self.extract_audio_features(paths["sep-28k"], paths["sep-28k-E"])
        label = self.process_label(clip_data)
        return {
            "clip_id": clipid,
            "audio_inputs": audio, # dict[str, tensor]
            **label,
        }

    
class FrameLevelAudioDataset(AudioDataset):
    def __init__(self, 
                 root, 
                 annotator,
                 symptom_names,
                 sampling_rate = 16000,
                 split = "train", 
                 label = None,
                 do_single_label = False,
                 do_speaker_adaptation = False,
                 classification_task = "multilabel",
                 feature_extractor: str = "wav2vec2",
                 pretrained_model_name: str = 'facebook/wav2vec2-base-960h',
                 **kwargs):

        super().__init__(root, annotator, sampling_rate, split, label, **kwargs)
        self.feature_extractor = feature_extractor
        self.pretrained_model_name = pretrained_model_name
        self.target_sampling_rate = sampling_rate
        self.classification_task = classification_task
        self.symptom_names = symptom_names
        self.do_speaker_adaptation = do_speaker_adaptation
        
        # multilabelかmulticlassで正解ラベルが変わる
        if self.classification_task == "multilabel":
            self.labels = ['P', 'B', 'SR', 'ISR', 'MUR', 'any']
        if self.classification_task == "multiclass":
            self.labels = ['P', 'B', 'SR', 'ISR', 'MUR', 'no-stutter', 'other']
        if self.classification_task == "binaryclass":
            self.labels = ["any", "no-stutter"]
            
        self.do_single_label = do_single_label
        if self.do_single_label == True:
            print("used do_single_label")
            self.data_df = self.prep_df_single()
            self.clips = self.data_df['clip_id'].reset_index(drop=True)
        else: print("not used do_single_label")

        # _getitem_で重複して行を取得しないために，clipsは一意な行のみを取得している
        self.clips = (self.data_df["clip_id"].drop_duplicates().reset_index(drop=True))

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
        
        
    def prep_df_single(self):
        # prep_dfに単ラベル分類を行う際の処理を追加
        # 行単位でparseし，症状が一種類かどうかを判定
        
        
        if self.classification_task == "multiclass" or self.classification_task == "binaryclass":
            raise ValueError("multiclass分類タスクでは，複数ラベルデータを引き受けるのが前提のため，prep_df_singleを用いるのは適切ではありません．")
        
        label_path = os.path.join(self.root, "labels", f"{self.annotator}.csv")
        df = pd.read_csv(label_path)
        # 任意のsplitのデータのみを取り出してラベルを取り出す
        df = df[df['split'] == self.split].reset_index(drop=True)
        
        if (self.split == "test") or (self.split == "val") or (self.split == "train"):
            label_list = [label for label in self.labels if label != "any"] # "any"のみを除外
            single_label_df = df[
                df[label_list].sum(axis=1) == 1
            ]
            return single_label_df
        
        if self.label: # 単ラベル分類時
            # labelがtrainで定義されてない場合は遷移しない
    
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
        
    def __getitem__(self, idx):
        
        clip_id = self.clips.iloc[idx]

        # clip_idに含まれる全アノテーション行を参照
        clip_rows = self.data_df[
            self.data_df["clip_id"] == clip_id
        ]

        # ファイル情報はクリップ内で共通なので先頭行を使用
        clip_data = clip_rows.iloc[0]
        media_file = clip_data['media_file']
        task = clip_data['task']
        audio_path = self.get_item_paths(f"{task}_{media_file}")['audio']

        if os.path.exists(audio_path):
            audio = self.read_h5(audio_path, clip_id)
        else:
            print(f"Audio file not found for clip {clip_id}")
            audio = torch.zeros(1, self.target_sampling_rate)  # Default to empty audio

        label = self.process_annotationlabel(clip_rows, self.feature_extractor, self.pretrained_model_name, self.target_sampling_rate) # 強ラベル作成

        return {
            "clip_id": clip_id,
            "audio_inputs": audio,
            **label, # 辞書の中身を展開: {"A": ~, "B": ~} -> {"A": ~}, {"B": ~} 
        }
    
    def process_annotationlabel(self, clip_rows, feature_extractor, pretrained_model_name, sr) -> Dict[str, list]:
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
        elif feature_extractor == "whisper":
            # whisperでは時刻フレームのつくり方が異なるため，正解データはwavlmと同じやり方で
            # 作成することとする．
            model_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        kernel_samples, stride_samples = self.calc_kernel_stride_samples(model_config.conv_kernel, model_config.conv_stride)

        clip_data = clip_rows.iloc[0]
        start_time = clip_data["start_time"]
        end_time = clip_data["end_time"]
        clip_duration = (end_time - start_time) # float
        clip_samples = int(clip_duration * sr)
        num_frames = (clip_samples - kernel_samples) // stride_samples + 1 # 3 sec -> 149frames
        kernel_sec = kernel_samples / sr
        stride_sec = stride_samples / sr

        # 症状名を格納したリストを作成
        # symptom_names = [name for name in self.labels if name not in ("any", "no-stutter", "other")]
        symptom_labels = {name: [0] * num_frames for name in self.symptom_names} # 各症状を0行列で初期化

        # 各アノテーション区間を，その行で陽性の症状にだけ割り当てる
        for _, row in clip_rows.iterrows():
            annotation_start = row["annotation_start"]
            annotation_end = row["annotation_end"]

            if pd.isna(annotation_start) or pd.isna(annotation_end):
                continue

            if annotation_end <= annotation_start: # 開始が終了よりも遅かった場合error出力
                raise ValueError(
                    f"Invalid annotation interval: "
                    f"{annotation_start}, {annotation_end}"
                )

            interval_labels = self.make_stronglabel(
                num_frames,
                start_time,
                end_time,
                annotation_start,
                annotation_end,
                kernel_sec,
                stride_sec,
            )
            
            for name in self.symptom_names:
                if row[name] == 1:
                    #既存区間とのOR．離れた区間の隙間は埋めない
                    symptom_labels[name] = [
                        int(old or new) # old or newで配列の各要素ごとにorをとる
                        for old, new in zip(symptom_labels[name], interval_labels)
                    ]

        if self.classification_task == "multilabel":
            labels = symptom_labels
            #anyは5症状のフレーム毎のOR
            labels["any"] = [
                int(any(values))
                for values in zip(*(symptom_labels[name] for name in self.symptom_names)) # label[name]は配列，zip()は同じフレームの値をまとめて取り出す
            ]
            
        elif self.classification_task == "multiclass":
            # 返すラベルは7クラス．各フレームで必ず1クラスだけ陽性とする
            class_names = [*self.symptom_names, "no-stutter", "other",]
            labels = {name: [0] * num_frames for name in class_names}
            
            for frame_idx in range(num_frames):
                active_symptoms = [
                    name for name in self.symptom_names if symptom_labels[name][frame_idx] == 1
                ] # 陽性のフレームがあるクラスから一フレームだけ取り出す

                if len(active_symptoms) == 0:
                    target_name = "no-stutter"
                elif len(active_symptoms) == 1:
                    target_name = active_symptoms[0]
                else:
                    target_name = "other"
                
                labels[target_name][frame_idx] = 1

        elif self.classification_task == "binaryclass":
            any_labels = [
                int(any(values))
                for values in zip(*(symptom_labels[name] for name in self.symptom_names))
            ]
            labels = {
                "any": any_labels,
                "no-stutter": [1 - value for value in any_labels], # no-stutterラベルはanyを反転させたもの
            }

        else:
            raise ValueError(f"Unknown classification_task: {self.classification_task}")
        
        return labels
    
    def make_stronglabel(self, 
                         num_frames: int,
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
                 classification_task: str = "multilabel",
                 split: str = "train", 
                 modality: Optional[str] = 'audio', 
                 feature_extractor: str = "wav2vec2", 
                 pretrained_model_name: str = 'facebook/wav2vec2-base-960h') -> Union[AudioDataset, VideoDataset, VideoAudioDataset]:

    if modality == "audio":
        return AudioDataset(**vars(config), split=split)
    elif modality == "exp_audio":
        return FrameLevelAudioDataset(**vars(config),
                                      classification_task=classification_task,
                                      split=split,
                                      feature_extractor=feature_extractor,
                                      pretrained_model_name=pretrained_model_name
                                      )
    elif modality == "pretrain":
        return Sep28kAudioDataset(**vars(config),
                                  classification_task=classification_task,
                                  split=split,
                                  )
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