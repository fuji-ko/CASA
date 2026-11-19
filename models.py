import sys
import numpy as np
from copy import deepcopy
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Wav2Vec2Model, VivitModel, WavLMModel, WhisperModel, HubertModel
from typing import Dict, List, Optional, Union, Tuple, Any
import torchmetrics
from abc import ABC, abstractmethod
from sklearn.metrics import precision_recall_curve
from configs import AudioModelConfig, VideoModelConfig, MultimodalModelConfig, BaseModelConfig, TrainingConfig
from huggingface_hub import hf_hub_download
from event_base_f1 import *
from architecture import *

ARCHITECTURES = {
    "lstm_singlehead": LSTMSingleHead,
    "lstm_multihead": LSTMMultiHead,
    "ensamble_transformer": TransformerEncoder,
    "ensamble_linear": LinearEnsambleHead,
    "ensamble_lstm": LstmEnsambleHead,
}

def _masked_mean(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute mean pooling with attention mask
    可変長の系列を，paddingを除外しながら1本のD次元ベクトルにする処理
    paddingした部分は平均計算に含めない"""
    # Extend mask to same shape as hidden states
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_sum = torch.sum(hidden_states * mask, dim=1)
    mask_sum = torch.sum(mask, dim=1)
    mask_sum = torch.clamp(mask_sum, min=1e-9)
    
    return masked_sum / mask_sum

class BaseModule(pl.LightningModule, ABC):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a model from the Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path (str): Model name or path.
            **kwargs: Additional arguments for the model.
        
        Returns:
            An instance of the model class.
        """
        # Download the model files
        model_ckpt = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename="model_best.ckpt",
            subfolder="checkpoint",
        )
        model = cls.load_from_checkpoint(model_ckpt, **kwargs)

        return model
 
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        # self.save_hyperparameters(vars(config))
        
        # Set label names
        self.label_names = config.label_names
        
        # These will be initialized by child classes
        self.backbone = None
        self.classifier = None
        
        self.train_accuracy = torchmetrics.Accuracy(task='binary', num_labels=len(config.label_names))
        self.val_accuracy = torchmetrics.Accuracy(task='binary', num_labels=len(config.label_names))
        self.test_accuracy = torchmetrics.Accuracy(task='binary', num_labels=len(config.label_names))
        
        self.train_f1 = torchmetrics.F1Score(task='binary', num_labels=len(config.label_names))
        self.val_f1 = torchmetrics.F1Score(task='binary', num_labels=len(config.label_names)) # calced micro F1
        # calced micro F1
        # 5種類のラベルを全部まとめて一つのconfusion matrixにしてF1を計算
        self.test_f1 = torchmetrics.F1Score(task='binary', num_labels=len(config.label_names)) # calced micro F1

        self.per_class_metrics = nn.ModuleDict()
        for label in self.label_names:
            self.per_class_metrics[f"val/f1_{label}"] = torchmetrics.F1Score(task='binary', num_classes=1)
            self.per_class_metrics[f"test/f1_{label}"] = torchmetrics.F1Score(task='binary', num_classes=1)

        
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.scheduler_eta_min = config.scheduler_eta_min
    
    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        pass
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} of {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def _get_labels_tensor(self, batch: Dict) -> torch.Tensor:
        """Convert batch labels to a tensor"""
        batch_size = len(batch["clip_id"])
        labels = torch.zeros(batch_size, len(self.label_names), device=self.device)
        for i, label in enumerate(self.label_names):
            if label in batch:
                # Convert label values to float
                label_values = batch[label]
                for j, val in enumerate(label_values):
                    if val is not None:
                        labels[j, i] = float(val)
        return labels
    
    def _common_step(self, batch, batch_idx, step_type):
        """Common step for training, validation and testing"""
        # Get inputs and labels based on child implementation
        inputs, labels = self._prepare_batch(batch)
        
        # Forward pass
        outputs = self(**inputs)
        logits = outputs["logits"] # (B, T, C)
        
        # Compute loss
        loss_weights = None
        if self.config.loss_weights is not None:
            loss_weights = torch.tensor(self.config.loss_weights, device=self.device)
            
        loss = F.binary_cross_entropy_with_logits(logits, labels, loss_weights, reduction='mean')

        # Compute predictions
        preds = torch.sigmoid(logits) > 0.5

        # Update metrics
        if step_type == "train":
            self.train_accuracy(preds, labels)
            self.train_f1(preds, labels)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=True)
            self.log("train/f1", self.train_f1, prog_bar=True, on_epoch=True)
        elif step_type == "val":
            self.val_accuracy(preds, labels)
            self.val_f1(preds, labels)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/acc", self.val_accuracy, prog_bar=True, on_epoch=True)
            self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
            self._update_per_class_metrics(preds, labels, step_type)
        elif step_type == "test":
            self.test_accuracy(preds, labels)
            self.test_f1(preds, labels)
            self.log("test/loss", loss, prog_bar=True, on_epoch=True)
            self.log("test/acc", self.test_accuracy, prog_bar=True, on_epoch=True)
            self.log("test/f1", self.test_f1, prog_bar=True, on_epoch=True) # macro f1
            self._update_per_class_metrics(preds, labels, step_type)
        
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels
        }
    
    def _update_per_class_metrics(self, preds, labels, step_type):
        """Helper method to update per-class metrics"""
        for i, label in enumerate(self.label_names):
            class_preds = preds[:, i]
            class_labels = labels[:, i]
            
            # Skip empty classes (all labels are 0)
            if class_labels.sum() > 0:
                # Update F1
                self.per_class_metrics[f"{step_type}/f1_{label}"](class_preds, class_labels)
                self.log(
                    f"{step_type}/f1_{label}", 
                    self.per_class_metrics[f"{step_type}/f1_{label}"], 
                    on_epoch=True
                )
                
                # # Update Accuracy
                # self.per_class_metrics[f"{step_type}_acc_{label}"](class_preds, class_labels)
                # self.log(
                #     f"{step_type}_acc_{label}", 
                #     self.per_class_metrics[f"{step_type}_acc_{label}"], 
                #     on_epoch=True
                # )

    @abstractmethod
    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pass
    
    def training_step(self, batch, batch_idx):
        """trainer.fitを呼び出すと一バッチごとにこの関数が呼び出される"""
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.scheduler_eta_min
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]
    
    def predict_stuttering(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict]:

        self.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self(**self._prepare_prediction_inputs(inputs))
            logits = outputs["logits"]
            
            # Compute probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
        
        results = {}
        for i, label in enumerate(self.label_names):
            results[label] = {
                "probability": probs[0, i].item(),
                "prediction": preds[0, i].item()
            }

        
        return results
    
    @abstractmethod
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

class AudioClassificationModule(BaseModule):
    """Wav2Vec2-based stuttering classification model"""

    def __init__(self, config: AudioModelConfig):
        super().__init__(config)
        
        # Load pre-trained wav2vec model
        self.backbone = Wav2Vec2Model.from_pretrained(config.pretrained_model_name)
        
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze feature extractor if needed
        if config.freeze_feature_extractor:
            self.freeze_feature_extraction()
        
        # Freeze encoder if needed
        if config.freeze_encoder:
            self.freeze_encoder()
            
        # Unfreeze specific layers if requested
        if config.unfreeze_layers:
            self.unfreeze_layers(config.unfreeze_layers)

        # create separate classification heads for each label
        if config.label_names is None or len(config.label_names) == 0:
            raise ValueError("label_names must be provided and non-empty for audio model")
        
        self.classifier = nn.ModuleDict()
        for label in self.label_names:
            self.classifier[label] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, 1)  # Binary classification for each label
            ) # nn.Sequentialは処理が上から下へ順番に流れる場合に，複数の層をまとめて書く関数
        
        self.print_trainable_parameters()
    
    def freeze_feature_extraction(self):
        """Freeze the feature extraction part of wav2vec2"""
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False
    
    def freeze_encoder(self):
        """Freeze the transformer encoder part of wav2vec2"""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_ids: List[int]):
        """Unfreeze specific encoder layers for fine-tuning"""
        for layer_id in layer_ids:
            for param in self.backbone.encoder.layers[layer_id].parameters():
                param.requires_grad = True
    
    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for audio model"""

        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, D)
        
        # Pool hidden states
        if attention_mask is not None:
            # Masked mean pooling
            pooled = self._masked_mean(hidden_states, attention_mask)
        else:
            # Global mean pooling
            pooled = hidden_states.mean(dim=1)  # (B, D)
        
        # Classification head
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(pooled)
        # Convert logits to a single tensor
        logits = torch.cat([logits[label] for label in self.label_names], dim=1) # shape (B, num_labels)
        
        return {
            "logits": logits,
            "pooled_output": pooled,
            "hidden_states": outputs.hidden_states
        }

    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare audio inputs and labels from batch"""
        audio_inputs = batch["audio_inputs"] # (B, 1, T)
        input_values = audio_inputs["input_values"].squeeze(1)  # (B, T)
        attention_mask = audio_inputs.get("attention_mask", None)
        
        inputs = {"input_values": input_values}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
            
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_values = inputs["input_values"].to(self.device)
        
        prepared_inputs = {"input_values": input_values}
        if "attention_mask" in inputs:
            prepared_inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            
        return prepared_inputs

class Sep28kAudioClassificationModule(BaseModule):
    """sep28k学習時のモデル"""
    def __init__(self, config: AudioModelConfig):
        super().__init__(config)
        self.feature_extractor = config.feature_extractor
        self.classification_task = config.classification_task

        # Load pre-trained feature_extractor model
        if self.feature_extractor == "wav2vec2":
            self.backbone = Wav2Vec2Model.from_pretrained(config.pretrained_model_name)
            print("used of wav2vec2")
        elif self.feature_extractor == "wavlm":
            self.backbone = WavLMModel.from_pretrained(config.pretrained_model_name) # 1028
            print("used of wavlm")
        elif self.feature_extractor == "whisper":
            self.backbone = WhisperModel.from_pretrained(config.pretrained_model_name)
            print("used of whisper")
        else:
            raise ValueError(f"Undefined feature_extractor: {self.feature_extractor}")

        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size

        # Freeze feature extractor if needed
        if config.freeze_feature_extractor:
            self.freeze_feature_extraction()
        
        # Freeze encoder if needed
        if config.freeze_encoder:
            self.freeze_encoder()
            
        # Unfreeze specific layers if requested
        if config.unfreeze_layers:
            self.unfreeze_layers(config.unfreeze_layers)

        # create separate classification heads for each label
        if config.label_names is None or len(config.label_names) == 0:
            raise ValueError("label_names must be provided and non-empty for audio model")

        # select classification model
        architecture_cls = ARCHITECTURES[config.architecture]
        self.prediction_model = architecture_cls(
            hidden_size = hidden_size,
            label_names=self.label_names,
            dropout=config.dropout,
        )
        self.print_trainable_parameters()

        self.test_conf = torchmetrics.classification.MultilabelConfusionMatrix(num_labels=len(self.label_names), threshold=0.5)
        if self.classification_task in ("multiclass", "binaryclass"):
            num_classes = len(self.label_names)
            
            self.train_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="micro"
            )
            self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
            )
            self.test_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="micro"
            )

            self.val_f1 = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="micro"
            )
            self.test_f1 = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="micro"
            )
            self.val_epoch_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes,average="micro")

    def freeze_feature_extraction(self):
        """Freeze the feature extraction part of wav2vec2
        whisperの場合は，conv層の凍結処理"""

        # whisperの場合はconv層の凍結処理とする
        if self.feature_extractor == "whisper":
            modules = [self.backbone.encoder.conv1,
                       self.backbone.encoder.conv2,
                       ]
        
        else: # wav2vec2 or wavlm
            modules = [self.backbone.feature_extractor]

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
    
    def freeze_encoder(self):
        """Freeze the transformer encoder part of wav2vec2"""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_ids: List[int]):
        """Unfreeze specific encoder layers for fine-tuning"""
        for layer_id in layer_ids:
            for param in self.backbone.encoder.layers[layer_id].parameters():
                param.requires_grad = True

    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for audio model"""

        if self.feature_extractor == "whisper":
            outputs = self.backbone.encoder(
                input_features=input_values,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state  # (B, T, D)
            # 正解ラベルとそろえるために，先頭149フレームまでを切り出す．
            hidden_states = hidden_states[:, :149, :]  # (B, 149, D)

            # wav2vec2, wavlm
        else:
            outputs = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.last_hidden_state  # (B, T, D)
        
        # classification head
        logits = self.prediction_model(hidden_states) # (B, T, C)
        
        # pooling
        if attention_mask is not None:
            pooled = self._masked_mean(logits, attention_mask)
        else:
            pooled = logits.mean(dim=1) # (B, C)
        
        return {
            "logits": logits,
            "pooled_output": pooled,
            "hidden_states": outputs.hidden_states
        }
        
    def _common_step(self, batch, batch_idx, step_type):
        """common step for training, validation."""
        inputs, labels = self._prepare_batch(batch)
        outputs = self(**inputs)
        logits = outputs["pooled_output"] #(B, T, C)
        
        # Compute loss weights
        loss_weights = None
        if self.config.loss_weights is not None:
            loss_weights = torch.tensor(self.config.loss_weights, device=self.device)
            
        # loss計算に使用する関数の選択
        # Sigmoidの場合は独立に予測確率を算出するので，BCEを使用
        if self.classification_task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(logits, labels, loss_weights, reduction="mean")
        # multiclassでは1クラス予測
        if self.classification_task == "multiclass" or self.classification_task == "binaryclass":
            # 各フレームで正解クラスが必ず一つであることを確認
            if not torch.all(labels.sum(dim=-1) == 1): # 各フレームのCをすべて加算, torch.allは全b, Tについて確認
                raise ValueError("multiclassでは各フレームの正解クラスを一つにして下さい")            

            targets = labels.argmax(dim=-1).long() # (B), Cについて．最大値を持つ位置を返す
            loss = F.cross_entropy(logits, targets, reduction="mean")

        if step_type =="train":
            preds = torch.sigmoid(logits) > 0.5 
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        elif step_type == "val":
            if self.classification_task == "multilabel":
                preds = torch.sigmoid(logits) > 0.5
                self.val_accuracy(preds, labels)
                self.val_f1(preds, labels)
                self.log("val/loss", loss, prog_bar=True, on_epoch=True)
                self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
                self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
                self._update_per_class_metrics(preds, labels, step_type)
            elif self.classification_task in ("multiclass", "binaryclass"):
                probs = torch.softmax(logits, dim=-1)
                pred_classes = probs.argmax(dim=-1)
                label_classes = labels.argmax(dim=-1).long()
                self.val_accuracy(pred_classes, label_classes)
                self.val_f1(pred_classes, label_classes)
                self.log("val/loss", loss, prog_bar=True, on_epoch=True)
                self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
                self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
                preds = F.one_hot(pred_classes, num_classes=logits.shape[-1]).bool()
                self._update_per_class_metrics(preds, labels, step_type)

        return {
            "loss": loss,
            "preds": preds,
            "labels": labels
        }
    
    def _update_per_class_metrics(self, preds, labels, step_type):
            """Helper method to update per-class metrics
            フレームレベル用
            元実装では，正解ラベルに陽性が入っていないものは評価から外す実装をしているが，これはおそらく不適切だろう"""

            for i, label in enumerate(self.label_names):
                class_preds = preds[:,i]
                class_labels = labels[:,i]
                
                # Skip empty classes (all labels are 0)
                # if class_labels.sum() > 0: <- これは入れるべきか議論すべき？？
                # Update F1
                self.per_class_metrics[f"{step_type}/f1_{label}"](class_preds, class_labels)
                self.log(
                    f"{step_type}/f1_{label}", 
                    self.per_class_metrics[f"{step_type}/f1_{label}"], 
                    on_epoch=True
                )

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.scheduler_eta_min
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]

    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare audio inputs and labels from batch"""
        audio_inputs = batch["audio_inputs"] # (B, 1, T)
        input_values = audio_inputs["input_values"].squeeze(1)  # (B, T)
        attention_mask = audio_inputs.get("attention_mask", None)
        
        inputs = {"input_values": input_values}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
            
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_values = inputs["input_values"].to(self.device)
        
        prepared_inputs = {"input_values": input_values}
        if "attention_mask" in inputs:
            prepared_inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            
        return prepared_inputs

    
class FrameLevelAudioClassificationModule(BaseModule):
    """frame-lebel stuttering classification model
    stuttering detectionで使用したモデル構造を再現"""

    def __init__(self, config: AudioModelConfig):
        super().__init__(config)
        self.feature_extractor = config.feature_extractor
        self.iou_threshold = config.iou_threshold
        self.do_median_filter = config.do_median_filter
        self.classification_task = config.classification_task
        
        # Load pre-trained feature_extractor model
        if self.feature_extractor == "wav2vec2":
            self.backbone = Wav2Vec2Model.from_pretrained(config.pretrained_model_name)
            print("used of wav2vec2")
        elif self.feature_extractor == "wavlm":
            self.backbone = WavLMModel.from_pretrained(config.pretrained_model_name) # 1028
            # self.backbone = HubertModel.from_pretrained(config.pretrained_model_name) # used AS70
            print("used of wavlm")
        elif self.feature_extractor == "whisper":
            self.backbone = WhisperModel.from_pretrained(config.pretrained_model_name)
            print("used of whisper")
        else:
            raise ValueError(f"Undefined feature_extractor: {self.feature_extractor}")
        
        # Load finetuned model 
        if config.finetuned_model_path is not None:
            checkpoint = torch.load(config.finetuned_model_path, map_location="cpu", weights_only=False)
            state = checkpoint["state_dict"]
            
        # Extractor解凍
        #     self.backbone.load_state_dict({
        #         key.removeprefix("backbone."): value
        #         for key, value in state.items()
        #         if key.startswith("backbone.")
        #     })
            
        #     # self.backbone.load_state_dict(checkpoint["wavlm"]) # used AS70
        #     print("loaded finetuned extractor model.")
                
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze feature extractor if needed
        if config.freeze_feature_extractor:
            self.freeze_feature_extraction()
        
        # Freeze encoder if needed
        if config.freeze_encoder:
            self.freeze_encoder()
            
        # Unfreeze specific layers if requested
        if config.unfreeze_layers:
            self.unfreeze_layers(config.unfreeze_layers)

        # create separate classification heads for each label
        if config.label_names is None or len(config.label_names) == 0:
            raise ValueError("label_names must be provided and non-empty for audio model")

        # select classification model
        architecture_cls = ARCHITECTURES[config.architecture]
        self.prediction_model = architecture_cls(
                hidden_size = hidden_size,
                label_names=self.label_names,
                dropout=config.dropout,
                )
        
        # finetuned checkpoint activate to classifier model
        if config.finetuned_model_path is not None:
            print("used finetuned model to classifier.")
            if self.classification_task in ("multiclass", "binaryclass"):
                prediction_state = {
                    key.removeprefix("prediction_model."): value
                    for key, value in state.items()
                    if key.startswith("prediction_model.")
                }
                if not prediction_state:
                    raise ValueError("prediction_modelの重みが見つかりません")
                
                # 最終Linear層だけ除外
                excluded_keys = {
                    "classifier.4.weight",
                    "classifier.4.bias",
                }
                prediction_state = {
                    key: value
                    for key, value in prediction_state.items()
                    if key not in excluded_keys
                }
                result = self.prediction_model.load_state_dict(
                    prediction_state,
                    strict=False,
                )
                print("loaded fintuned classifier model.")
                
                # 意図した最終層以外にも不足，余分なキーがないか確認
                if (
                    set(result.missing_keys) != excluded_keys
                    or result.unexpected_keys
                ):
                    raise RuntimeError(f"想定外の重み不一致: {result}")
            elif self.classification_task == "multilabel":
                # stateはbackboneに取得したcheckpoint["state_dict"]を使用
                self.load_sep28k_multihead_weights(
                    prediction_model=self.prediction_model,
                    checkpoint_state=state,
                )
                
            
        self.print_trainable_parameters()
        
        # define event metrics
        self.event_metrics = {"val":{},
                            "test":{}
                                }
        for label in self.label_names:
            self.event_metrics["test"][label] = {"tp":0, "fp":0, "fn":0}
            # val/event_f1については，epochごとに出力
            self.event_metrics["val"][label] = {"tp": 0, "fp": 0, "fn": 0}
        # memo
        # torchmetrics.F1Scoreを定義することで,stepごとのTP, FP, TNなどの統計量を蓄積し，epoch終了時にまとめてF1を再計算
        
        # epochごとにaccuracyとf1を計算
        # 全バッチで計算した閾値で求めた予測クラスを使いたいため
        self.val_epoch_accuracy = torchmetrics.Accuracy(task="binary", num_labels=len(self.label_names))
        self.val_epoch_f1 = torchmetrics.F1Score(task="binary", num_labels=len(self.label_names))
        # フレームレベルスコアの混同行列統計量
        self.test_conf = torchmetrics.classification.MultilabelConfusionMatrix(num_labels=len(self.label_names), threshold=0.5)

        if self.classification_task in ("multiclass", "binaryclass"):
            num_classes = len(self.label_names)
            
            self.train_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="micro"
            )
            self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
            )
            self.test_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="micro"
            )

            self.val_f1 = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="micro"
            )
            self.test_f1 = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="micro"
            )
            self.val_epoch_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes,average="micro")
            
    def load_sep28k_multihead_weights(self,
                                 prediction_model: LSTMMultiHead,
                                 checkpoint_state: Dict[str, torch.Tensor]):
        """sep28kのmultihead重みをCASAへ転用する"""
        if not isinstance(prediction_model, LSTMMultiHead):
            raise TypeError("LSTMMultiHeadを指定してください")
        
        # casa -> sep
        head_mapping = {
            "ISR": "SoundRep",
            "SR": "SoundRep",
            "MUR": "WordRep",
            "B": "Block",
            "P": "Prolongation",
            "any": "any",
        }
        if set(prediction_model.label_names) != set(head_mapping):
            raise ValueError(f"想定外のCASAラベル: {prediction_model.label_names}")
        
        prefix = "prediction_model."
        source_state = {
            key.removeprefix(prefix): value
            for key, value in checkpoint_state.items()
            if key.startswith(prefix)
            }
        
        if not source_state:
            raise ValueError("sep分類器の重みが見つかりません")
        
        target_state = prediction_model.state_dict()
        transfer_state = {}
        
        # 各headの最終linearは転用しなくてもいいが，今回は転用することにする
        excluded_keys = {
            f"classifier.{label}.4.{parameter}"
            for label in prediction_model.label_names
            for parameter in ("weight", "bias")
        }
        
        for target_key, target_value in target_state.items():
            # if target_key in excluded_keys:
            #     continue
            
            if target_key.startswith("lstm."):
                source_key = target_key
                
            elif target_key.startswith("classifier."):
                _, target_label, suffix = target_key.split(".", 2)
                source_label = head_mapping[target_label]
                source_key = f"classifier.{source_label}.{suffix}"
                
            else:
                raise ValueError(f"未対応のパラメータ: {target_key}")
            
            if source_key not in source_state:
                raise KeyError(f"転用元の重みがありません: {source_key}")
            
            source_value = source_state[source_key]
            
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"形状不一致: {source_key} → {target_key}: "
                    f"{tuple(source_value.shape)} != "
                    f"{tuple(target_value.shape)}"
                )
                
            transfer_state[target_key] = source_value
            
        result = prediction_model.load_state_dict(transfer_state,strict=True)
        if result.unexpected_keys:
            raise RuntimeError(f"想定外の重み不一致: {result}")
        return result
            
    def freeze_feature_extraction(self):
        """Freeze the feature extraction p1art of wav2vec2
        wavlm系の場合は，全層凍結
        whisperの場合は，conv層の凍結処理"""

        # whisperの場合はconv層の凍結処理とする
        if self.feature_extractor == "whisper":
            modules = [self.backbone.encoder.conv1,
                       self.backbone.encoder.conv2,
                       ]
        
        else: # wav2vec2 or wavlm
            # modules = [self.backbone.feature_extractor] #TODO
            modules = [self.backbone]

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
    
    def freeze_encoder(self):
        """Freeze the transformer encoder part of wav2vec2"""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_ids: List[int]):
        """Unfreeze specific encoder layers for fine-tuning"""
        for layer_id in layer_ids:
            for param in self.backbone.encoder.layers[layer_id].parameters():
                param.requires_grad = True
    
    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for audio model"""

        if self.feature_extractor == "whisper":
            outputs = self.backbone.encoder(
                input_features=input_values,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state  # (B, T, D)
            # 正解ラベルとそろえるために，先頭149フレームまでを切り出す．
            hidden_states = hidden_states[:, :149, :]  # (B, 149, D)

            # wav2vec2, wavlm
        else:
            outputs = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.last_hidden_state  # (B, T, D)
        
        logits = self.prediction_model(hidden_states)
        
        return {
            "logits": logits,
            "pooled_output": None,
            "hidden_states": outputs.hidden_states
        }

    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare audio inputs and labels from batch"""
        audio_inputs = batch["audio_inputs"] # (B, 1, T)

        if self.feature_extractor == "whisper":
            inputs = {"input_values": audio_inputs["input_features"].squeeze(1)}

        else:
            input_values = audio_inputs["input_values"].squeeze(1)  # (B, T)
            attention_mask = audio_inputs.get("attention_mask", None)
            
            inputs = {"input_values": input_values}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            
        labels = self._get_framelabels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_values = inputs["input_values"].to(self.device)
        
        prepared_inputs = {"input_values": input_values}
        if "attention_mask" in inputs:
            prepared_inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            
        return prepared_inputs
    
    def _get_framelabels_tensor(self, batch: Dict) -> torch.Tensor:
        """convert batch framelabels to a tensor(list)"""
        batch_size = len(batch["clip_id"])
        num_frames = len(batch[self.label_names[0]][0])
        
        labels = torch.zeros(batch_size, num_frames, len(self.label_names), device=self.device, dtype=torch.float32)
        for i, label in enumerate(self.label_names):
            if label in batch:
                # Convert label values to float
                label_values = batch[label]
                for j, val in enumerate(label_values):
                    if val is not None:
                        labels[j, :, i] = torch.tensor(val, dtype=torch.float32, device=self.device)
        return labels # (B, T, labels)

    def _common_step(self, batch, batch_idx, step_type):
        """
        フレームレベル検出用関数
        Common step for training, validation and testing"""
        # Get inputs and labels based on child implementation
        inputs, labels = self._prepare_batch(batch)
        
        # Forward pass
        outputs = self(**inputs)
        logits = outputs["logits"] # (B, T, C)
        
        # Compute loss weights
        loss_weights = None
        if self.config.loss_weights is not None:
            loss_weights = torch.tensor(self.config.loss_weights, device=self.device)
            
        # loss計算に使用する関数の選択
        # sigmoidの場合は独立に予測確率を算出するので，BCEを使用
        if self.classification_task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(logits, labels, loss_weights, reduction='mean')
        # multiclassでは一クラスを確実に予測するのでcross entropy lossを選択
        if self.classification_task == "multiclass" or self.classification_task == "binaryclass":
            # 各フレームで正解クラスが必ず一つであることを確認
            if not torch.all(labels.sum(dim=-1) == 1): # 各フレームのCをすべて加算, torch.allは全b, Tについて確認
                raise ValueError("multiclassでは各フレームの正解クラスを一つにして下さい")
            
            targets = labels.argmax(dim=-1).long() # (B,T) Cについて，最大値を持つ「位置」を返す
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), # (B*T, C)
                targets.reshape(-1), # (B*T,)
                reduction="mean",
            )

        # # Compute predictions
        # preds = torch.sigmoid(logits) > 0.5

        # Update metrics
        if step_type == "train":
            # self.train_accuracy(preds, labels)
            # self.train_f1(preds, labels)
            preds = torch.sigmoid(logits) > 0.5
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            # self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=True)
            # self.log("train/f1", self.train_f1, prog_bar=True, on_epoch=True)
            
        elif step_type == "val":
            if self.classification_task == "multilabel":
                probs = torch.sigmoid(logits)

                # stepごとに閾値を計算して正解率とf1を算出
                # 煩雑なため，あまり参考にならない可能性あり
                best_th_list = self.return_best_f1_threshold(labels.detach(), probs.detach())
                preds = probs > torch.tensor(best_th_list, device=self.device)
                self.val_accuracy(preds, labels)
                self.val_f1(preds, labels)
                
            elif self.classification_task in ("multiclass", "binaryclass"):
                probs = torch.softmax(logits, dim=-1)
                pred_classes  = probs.argmax(dim=-1) # (B,T)
                label_classes = labels.argmax(dim=-1).long()
                
                self.val_accuracy(pred_classes, label_classes)
                self.val_f1(pred_classes, label_classes)
                
                # 既存の症状別F1, event評価に渡す形式へ戻す(per_class_metrics算出用)
                # C次元については，予測クラスの位置だけを1にし，それ以外は0にする
                preds = F.one_hot(pred_classes,
                                  num_classes=logits.shape[-1],
                                  ) # (B, T, C) bool()で1をTrue, 0をFalseに変換
                                            # pred_classes時点では，任意の数字が振られているため
                
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/acc", self.val_accuracy, prog_bar=True, on_epoch=True) # 
            self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
            self._update_per_class_metrics(preds, labels, step_type)
            
        elif step_type == "test":
            if self.classification_task == "multilabel":
                print(f'{self.best_thresholds=}')
                print(f'`{self.window_dict=}')
                preds = torch.sigmoid(logits) > torch.tensor(self.best_thresholds, device=self.device)
                # ここで平滑化処理
                if self.do_median_filter == True:
                    preds = self.apply_median_filter(preds, self.window_dict)
                self.test_accuracy(preds, labels)
                self.test_f1(preds, labels)

            elif self.classification_task in ("multiclass", "binaryclass"):
                probs = torch.softmax(logits, dim=-1)
                pred_classes = probs.argmax(dim=-1)
                label_classes = labels.argmax(dim=-1).long()

                self.test_accuracy(pred_classes, label_classes)
                self.test_f1(pred_classes, label_classes)
                
                preds = F.one_hot(
                    pred_classes,
                    num_classes=logits.shape[-1],
                ).bool()
            self.log("test/loss", loss, prog_bar=True, on_epoch=True)
            self.log("test/acc", self.test_accuracy, prog_bar=True, on_epoch=True)
            self.log("test/f1", self.test_f1, prog_bar=True, on_epoch=True)
            self._update_per_class_metrics(preds, labels, step_type)
        
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "logits": logits
        }

    def on_validation_epoch_start(self):
        """validation開始時にlightningが呼び出し"""
        self.val_labels = []
        self.val_probs = []

    def validation_step(self, batch, batch_idx):
        """commonではない処理を記述"""
        outputs = self._common_step(batch, batch_idx, "val")

        # 各ステップ計算結果を集計
        if self.classification_task == "multilabel":
            self.val_labels.append(outputs["labels"].detach())
            self.val_probs.append(torch.sigmoid(outputs["logits"]).detach())
        elif self.classification_task in ("multiclass", "binaryclass"):
            self.val_labels.append(outputs["labels"].detach())
            self.val_probs.append(outputs["preds"].detach())
        else: raise ValueError("予期しないtask名です")
        
        # event-based metrics
        val_counts = event_based_f1(preds=outputs["preds"],
                                    labels=outputs["labels"],
                                    label_names=self.label_names,
                                    iou_threshold=self.iou_threshold)
        
        # ここから下は不要説
        # 検証データの時点でevent-f1は求める必要なし．閾値処理が複雑であるため
        for c, label in enumerate(self.label_names):
            self.event_metrics["val"][label]["tp"] += val_counts[label]["tp"]
            self.event_metrics["val"][label]["fp"] += val_counts[label]["fp"]
            self.event_metrics["val"][label]["fn"] += val_counts[label]["fn"]

        return outputs

    def on_validation_epoch_end(self):
        # select best f1 threshold per class
        labels = torch.cat(self.val_labels, dim=0) # [(B, T, C), (B, T, C), ...] -> [(B*n, T, C)]
        probs = torch.cat(self.val_probs, dim=0)
        
        if self.classification_task == "multilabel":
            self.best_thresholds = self.return_best_f1_threshold(labels, probs)
            preds = probs > torch.tensor(self.best_thresholds, device=self.device)
            if self.do_median_filter == True:
                self.window_dict = make_filter_list(preds, labels, self.label_names, self.iou_threshold)
                preds = self.apply_median_filter(preds, self.window_dict)
        elif self.classification_task in ("multiclass", "binaryclass"):
            preds = probs # int型のone-hot vector
        else: raise ValueError("予期しないtask名です．")
        
        # 全バッチを集計したスコアを算出
        if self.classification_task in ("multiclass", "binaryclass"):
            self.val_epoch_accuracy(preds.long().argmax(dim=-1), labels.argmax(dim=-1).long()) #(B,T,C)->(B,T)
        else: self.val_epoch_accuracy(preds, labels)
        
        self.val_epoch_f1(preds, labels)
        self.log("val/epoch_acc", self.val_epoch_accuracy, prog_bar=True, on_epoch=True)
        self.log("val/epoch_f1", self.val_epoch_f1, prog_bar=True, on_epoch=True)
        
        # calc event-based score
        results = self._compute_event_f1("val")
        for label in self.label_names:
            f1 = results[label]["f1"]
            self.log(f"val/event_f1_{label}",
                     f1,
                     prog_bar=True,
                     sync_dist=True)
            
        self._reset_event_metrics("val")
        
    def on_save_checkpoint(self, checkpoint):
        """validation_epoch終了後に呼び出される"""
        print("debag: loaded on_save_checkpoint")
        super().on_save_checkpoint(checkpoint)
        if self.classification_task == "multilabel" and self.do_median_filter == True:
            checkpoint["median_k"] = self.window_dict
        if self.classification_task == "multilabel":
            checkpoint["best_thresholds"] = self.best_thresholds
        
    def on_load_checkpoint(self, checkpoint):
        """test開始時に呼び出される"""
        print(f"debag: loaded on_load_checkpoint")
        super().on_load_checkpoint(checkpoint)
        if self.classification_task == "multilabel" and self.do_median_filter == True:
            self.window_dict = checkpoint["median_k"]
        if self.classification_task == "multilabel":
            self.best_thresholds = checkpoint["best_thresholds"]
    
    def test_step(self, batch, batch_idx):
        outputs = self._common_step(batch, batch_idx, "test")
        # 混同行列要素を集計
        self.test_conf.update(
            outputs["preds"].reshape(-1, outputs["preds"].shape[-1]),
            outputs["labels"].reshape(-1, outputs["labels"].shape[-1]).int()
        ) # (B, T, C) => (B*T, C)
    
        #calc event-f1
        test_counts = event_based_f1(preds=outputs["preds"],
                                    labels=outputs["labels"],
                                    label_names=self.label_names,
                                    iou_threshold=self.iou_threshold)
        
        for c, label in enumerate(self.label_names):
            self.event_metrics["test"][label]["tp"] += test_counts[label]["tp"]
            self.event_metrics["test"][label]["fp"] += test_counts[label]["fp"]
            self.event_metrics["test"][label]["fn"] += test_counts[label]["fn"]
            
        return outputs
    
    def on_test_epoch_end(self):
        """event_scoreを出力"""
        results = self._compute_event_f1("test")
        for label in self.label_names:
            self.log(
                f"test/event_f1_{label}",
                results[label]["f1"],
                sync_dist=True
            )
        # 結果出力
        cm = self.test_conf.compute()  # (C, 2, 2)

        tn = cm[:, 0, 0]
        fp = cm[:, 0, 1]
        fn = cm[:, 1, 0]
        tp = cm[:, 1, 1]

        for i, name in enumerate(self.label_names):
            print(
                f"{name}: "
                f"TP={tp[i].item()}, "
                f"FP={fp[i].item()}, "
                f"FN={fn[i].item()}, "
                f"TN={tn[i].item()}"
            )
            
    def apply_median_filter(self,
                            preds: torch.Tensor,
                            window_dict: Dict[str, int]
                            ) -> torch.Tensor:
        """テスト時の予測データに検証データで選択したfilter値でmedian filterを適用"""
        preds = preds.detach().cpu().numpy()
        
        raw_preds = preds.copy()
        preds_med = np.zeros_like(raw_preds)
        B,T,C = raw_preds.shape
        for b in range(B):
            for c, label in enumerate(self.label_names):
                preds_med[b,:,c] = median_filter_1d(raw_preds[b,:,c], window_dict[label])
            
        return torch.from_numpy(preds_med).to(self.device)

    def return_best_f1_threshold(self,
                                 labels: torch.Tensor, # (B, T, C)
                                 logits: torch.Tensor # (B, T, C)
                                 ) -> list[int]:
        best_th = []
        labels = labels.flatten(0, 1).cpu().numpy()
        logits = logits.flatten(0, 1).cpu().numpy()
        for i in range(len(self.label_names)):
            label = labels[:,i]
            logit = logits[:,i]
            
            precision, recall, thresholds = precision_recall_curve(label, logit)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            best_idx = np.argmax(f1)
            best_threshold = thresholds[best_idx]
            best_th.append(best_threshold)
        return best_th

    def _update_per_class_metrics(self, preds, labels, step_type):
            """Helper method to update per-class metrics
            フレームレベル用
            元実装では，正解ラベルに陽性が入っていないものは評価から外す実装をしているが，これはおそらく不適切だろう"""

            for i, label in enumerate(self.label_names):
                class_preds = preds[:,:,i]
                class_labels = labels[:,:,i]
                
                # Skip empty classes (all labels are 0)
                # if class_labels.sum() > 0: <- これは入れるべきか議論すべき？？
                # Update F1
                self.per_class_metrics[f"{step_type}/f1_{label}"](class_preds, class_labels)
                self.log(
                    f"{step_type}/f1_{label}", 
                    self.per_class_metrics[f"{step_type}/f1_{label}"], 
                    on_epoch=True
                )
                
    def _compute_event_f1(self, stage: str) -> Dict[str, float]:
        results = {}
        
        for label in self.label_names:
            counts = self.event_metrics[stage][label]
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            
            precision = (tp / (tp + fp) if tp + fp > 0 else 0.0)
            recall = (tp / (tp + fn) if tp + fn > 0 else 0.0)
            if precision + recall > 0:
                f1 = (2 * precision * recall) / (precision + recall)
            else: f1 = 0.0
            
            results[label] = {"precision": precision, "recall": recall, "f1": f1}
        return results
    
    def _reset_event_metrics(self, stage):
        """validationのeventスコアをリセットする
        valごとにスコアを算出するため"""
        for label in self.label_names:
            self.event_metrics[stage][label] = {"tp":0, "fp":0, "fn":0}

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.scheduler_eta_min
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]
    
class EnsambleAudioClassificationModule(FrameLevelAudioClassificationModule):
    def __init__(self,
                 config: AudioModelConfig,
                 binary_checkpoint_path,
                 multiclass_checkpoint_path,
                 multilabel_checkpoint_path,
                 train_fronted: bool = False, # 前段モデルのclassifierを学習させるか
                 ):
        # アンサンブルモデルの後段を定義
        super().__init__(config)
        self.train_fronted = train_fronted
        hidden_size = self.backbone.config.hidden_size
        output_layer = config.output_classifier_layer
        
        frontends_definitions = [ # list[tuple]
            (
                "binary",
                binary_checkpoint_path,
                LSTMSingleHead,
                ["any", "no-stutter"],
            ),
            (
                "multiclass",
                multiclass_checkpoint_path,
                LSTMSingleHead,
                ["P", "B", "SR", "ISR", "MUR", "no-stutter", "other"],
            ),
            (
                "multilabel",
                multilabel_checkpoint_path,
                LSTMMultiHead,
                ["P", "B", "SR", "ISR", "MUR", "any"],
            ),
        ]
        
        def select_layers(head, layer_index):
            """指定した層までを取り出し，その出力次元も返す"""
            """classifierの構造が変われば，この関数は使えないことに注意"""
            if not isinstance(layer_index, int):
                raise ValueError("output_layer must be an integer")
            
            index = layer_index
            if index < 0: # -1や-2は末尾から数える
                index += len(head)
                
            if not 0 <= index < len(head):
                raise ValueError(
                    f"output_layer={layer_index} is out of range"
                    f"for a head with {len(head)} layers"
                )
            
            # 指定した番号までの層のnn.Sequentialが得られる
            selected = head[:index + 1] # output: nn.Sequential
            
            # 現在のheadではLinear以外の層は次元を変えない
            # 取り出した範囲内で最後のLinear層の出力次元を得る
            # GeLUやLayerNormは次元を変えないため
            for layer in reversed(list(selected)):
                if isinstance(layer, nn.Linear):
                    output_size = layer.out_features # 出力次元を取得
                    break
            
            return selected, output_size
        
        self.frontends = nn.ModuleDict()
        feature_size = 0
        
        for index, (name, path, classifier_cls, labels) in enumerate(frontends_definitions):
            # 独立したextractorを用意
            backbone = (self.backbone if index == 0 else deepcopy(self.backbone))
            
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint["state_dict"]
            backbone.load_state_dict({
                key.removeprefix("backbone."): value
                for key, value in state.items()
                if key.startswith("backbone.")
            })
            
            # 保存先と同じ構造で分類器を復元
            classifier = classifier_cls(
                hidden_size=hidden_size,
                label_names = labels,
                dropout = config.dropout
            )
            classifier.load_state_dict({
                key.removeprefix("prediction_model."): value
                for key, value in state.items()
                if key.startswith("prediction_model.")
            })
            
            layer_index = output_layer
            
            # singleheadもModuleDictに統一し，forwardで同じ処理にする
            heads = nn.ModuleDict()
            
            if isinstance(classifier, LSTMSingleHead):
                selected, size = select_layers(classifier.classifier, layer_index)
                heads["single"] = selected
                feature_size += size
                
            elif isinstance(classifier, LSTMMultiHead):
                for label in labels:
                    selected, size = select_layers(classifier.classifier[label], layer_index)
                    heads[label] = selected
                    feature_size += size
            
            # 各前段モデルから抽出したモデルを格納
            self.frontends[name] = nn.ModuleDict({
                "backbone": backbone,
                "lstm": classifier.lstm,
                "heads": heads
            })
            
            del classifier, state, checkpoint
        
        # 元の登録名を削除．extractorはfrontends内で保持される
        del self.backbone
        
        # 後段の入力次元を一定にする際に使用
        # self.feature_projection = nn.Linear(feature_size, hidden_size)
            
        # select classification model
        if config.architecture == "ensamble_transformer":
            if feature_size == 15: # 後段linear使用時
                nhead = 1
            else: nhead=8
            
            self.prediction_model = ARCHITECTURES[config.architecture](
            hidden_size = feature_size,
            label_names=self.label_names,
            dropout=config.dropout,
            nhead=nhead,
        )
        else: 
            self.prediction_model = ARCHITECTURES[config.architecture](
                hidden_size = feature_size,
                label_names=self.label_names,
                dropout=config.dropout
            )
        
        self.frontends.requires_grad_(train_fronted) # extractorもclassifierもまとめて凍結or学習
        
        # self.feature_projection.requires_grad_(True)
        
        self.prediction_model.requires_grad_(True)
        
        # 初期状態のtrain/valも設定
        self.train(self.training)
        
    def train(self, mode=True):
        super().train(mode)
        
        if not self.train_fronted:
            self.frontends.eval()
            
        return self

    def forward(self, 
                input_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        
        context = (nullcontext() if self.train_fronted else torch.no_grad())
        # self.train_fronted == Trueなら何も変更せず勾配計算が有効
        # Falseならtorch.no_gradが発動

        with context:
            features = []
            
            for branch in self.frontends.values():
                outputs = branch["backbone"](
                    input_values=input_values,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                lstm_features, _ = branch["lstm"](outputs.last_hidden_state)
                # features.append(lstm_feature)
                
                for head in branch["heads"].values():
                    features.append(head(lstm_features))
            
            combined_features = torch.cat(features, dim=-1) # (B, T, D*3)
            
        # features = self.feature_projection(combined_features) # (B, T, 1024)
        
        logits = self.prediction_model(combined_features)
    
        # 射影層は前段凍結時も学習するため，with contextの外で実行
        # 
        # projected_features = self.feature_projection(combined_features)
        # logits = self.prediction_model(projected_features)
    
        return {
            "logits": logits,
            "pooled_output": None,
            "hidden_states": combined_features
        }


class VideoClassificationModule(BaseModule):
    """ViVIT-based stuttering classification model"""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__(config)
        
        # Load pre-trained ViVIT model
        self.backbone = VivitModel.from_pretrained(config.pretrained_model_name)
        
        # Get hidden size from config
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze backbone if needed
        if config.freeze_backbone:
            self.freeze_backbone()
            
            # Optionally unfreeze the last N transformer layers
            if config.unfreeze_last_n_layers > 0:
                self.unfreeze_last_n_layers(config.unfreeze_last_n_layers)
        
        self.classifier = nn.ModuleDict()
        for label in config.label_names:
            # Create a separate classification head for each label
            self.classifier[label] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, 1)  # Binary classification for each label
            )

        
        self.print_trainable_parameters()
    
    def freeze_backbone(self):
        """Freeze the entire ViVIT backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last N transformer layers of ViVIT"""
        # Unfreeze specific layers (the last n layers)
        for i in range(len(self.backbone.encoder.layer) - n, len(self.backbone.encoder.layer)):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(
        self, 
        pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for video model"""
        # Get ViVIT outputs
        outputs = self.backbone(
            pixel_values=pixel_values.squeeze(1),  # (B, C, T, H, W) -> (B, T, C, H, W)
            output_hidden_states=True,
        )
        
        # Get the pooler output
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Classification head
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(pooled_output)
        
        logits = torch.cat([logits[label] for label in self.label_names], dim=1)
        
        return {
            "logits": logits,
            # "pooled_output": pooled_output,
            # "hidden_states": outputs.hidden_states
        }
    
    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare video inputs and labels from batch"""
        video_inputs = batch["video_inputs"]
        pixel_values = video_inputs["pixel_values"]
        
        inputs = {"pixel_values": pixel_values}
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare video inputs for prediction"""
        return {"pixel_values": inputs["pixel_values"].to(self.device)}
        
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.scheduler_eta_min
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]

class MultimodalClassificationModule(BaseModule):
    """
    Multimodal (audio+video) stuttering detection model
    
    This class combines both Wav2Vec2 and ViVIT models for multimodal analysis.
    """
    
    def __init__(self, config: MultimodalModelConfig):
        # We'll use the video config as the base since it typically has lower learning rate
        super().__init__(config)
        
        # Save both configs
        self.audio_config = config.audio_config
        self.video_config = config.video_config
        self.save_hyperparameters(vars(config))
        
        # Load pre-trained models
        self.audio_backbone = Wav2Vec2Model.from_pretrained(config.audio_config.pretrained_model_name)
        self.video_backbone = VivitModel.from_pretrained(config.video_config.pretrained_model_name)
        
        # Get hidden sizes
        audio_hidden_size = self.audio_backbone.config.hidden_size
        video_hidden_size = self.video_backbone.config.hidden_size
        combined_hidden_size = audio_hidden_size + video_hidden_size
        
        # Apply freezing strategies
        if config.audio_config.freeze_feature_extractor:
            for param in self.audio_backbone.feature_extractor.parameters():
                param.requires_grad = False
                
        if config.audio_config.freeze_encoder:
            for param in self.audio_backbone.encoder.parameters():
                param.requires_grad = False
                
        if config.video_config.freeze_backbone:
            for param in self.video_backbone.parameters():
                param.requires_grad = False
            
        # Selective unfreezing
        if config.audio_config.unfreeze_layers:
            for layer_id in config.audio_config.unfreeze_layers:
                for param in self.audio_backbone.encoder.layers[layer_id].parameters():
                    param.requires_grad = True
                    
        if config.video_config.unfreeze_last_n_layers > 0:
            for i in range(len(self.video_backbone.encoder.layer) - config.video_config.unfreeze_last_n_layers, 
                           len(self.video_backbone.encoder.layer)):
                for param in self.video_backbone.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_hidden_size, combined_hidden_size),
            nn.LayerNorm(combined_hidden_size),
            nn.GELU(),
            nn.Dropout(config.video_config.dropout)
        )
        
        # Classification head
        self.classifier = nn.ModuleDict()
        for label in config.label_names:
            self.classifier[label] = nn.Sequential(
                nn.Linear(combined_hidden_size, combined_hidden_size // 2),
                nn.LayerNorm(combined_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(config.video_config.dropout),
                nn.Linear(combined_hidden_size // 2, 1)
            )
        
        # Print parameter stats
        self.print_trainable_parameters()
    
    def forward(
        self, 
        input_values: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal model"""
        # Process audio
        audio_outputs = self.audio_backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Process video
        video_outputs = self.video_backbone(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        # Get audio pooled representation
        audio_hidden_states = audio_outputs.last_hidden_state
        # if attention_mask is not None:
        #     # Masked mean pooling
        #     audio_pooled = self._masked_mean(audio_hidden_states, attention_mask)
        # else:
        # Global mean pooling
        audio_pooled = audio_hidden_states.mean(dim=1)
    
        # Get video pooled representation
        video_pooled = video_outputs.pooler_output
        
        # Concatenate modalities
        multimodal_features = torch.cat([audio_pooled, video_pooled], dim=1)
        
        # Fusion
        fused_features = self.fusion(multimodal_features)
        
        # Classification
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(fused_features)
        # Convert logits to a single tensor
        logits = torch.cat([logits[label] for label in self.label_names], dim=1)
        
        return {
            "logits": logits,
            # "pooled_output": fused_features,
            # "audio_pooled": audio_pooled,
            # "video_pooled": video_pooled,
            # "audio_hidden_states": audio_outputs.hidden_states,
            # "video_hidden_states": video_outputs.hidden_states
        }
    
    def _prepare_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare multimodal inputs and labels from batch"""

        # Prepare inputs dict
        inputs = {
            "input_values": batch["audio_inputs"]["input_values"].squeeze(1),  # (B, T)
            "pixel_values": batch["video_inputs"]["pixel_values"].squeeze(1)  # (B, T, C, H, W)
        }
        
        inputs["attention_mask"] = batch["audio_inputs"].get(["attention_mask"], None)
            
        labels = self._get_labels_tensor(batch)
        
        return inputs, labels
    
    def _prepare_prediction_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare multimodal inputs for prediction"""
        prepared_inputs = {
            "input_values": inputs["audio_inputs"]["input_values"].to(self.device),
            "pixel_values": inputs["video_inputs"]["pixel_values"].to(self.device)
        }
        
        if "attention_mask" in inputs["audio_inputs"]:
            prepared_inputs["attention_mask"] = inputs["audio_inputs"]["attention_mask"].to(self.device)
            
        return prepared_inputs

def prep_model(config: TrainingConfig) -> BaseModule:

    if config.modality == "audio":
        config.audio_model_config.label_names = ['P', 'B', 'SR', 'ISR','MUR', 'any']
        return AudioClassificationModule(config.audio_model_config)
    
    elif config.modality == "exp_audio":
        #分類タスク方法を指定
        if config.audio_model_config.classification_task == "multilabel":
            config.audio_model_config.label_names = ['P', 'B', 'SR', 'ISR','MUR', 'any'] # 順番が大事
            if config.audio_model_config.architecture.split("_")[0] == "ensamble": # ensambleはarchitectureで指定
                return EnsambleAudioClassificationModule(config.audio_model_config,
                                                         binary_checkpoint_path="/work/abelab5/k_fuji/CASA/exp/wavlm_binaryclass_loaded_fintuned_model/checkpoints/stuttering-epoch=08-val_f1=0.00.ckpt",
                                                         multiclass_checkpoint_path="/work/abelab5/k_fuji/CASA/exp/wavlm_multiclass_loaded_fintuned_model/checkpoints/stuttering-epoch=10-val_f1=0.00.ckpt",
                                                         multilabel_checkpoint_path="/work/abelab5/k_fuji/CASA/exp/wavlm_multilabel_loaded_fintuned_model/checkpoints/stuttering-epoch=09-val_f1=0.00.ckpt",
                                                         train_fronted=False,
                                                         )
            
        elif config.audio_model_config.classification_task == "multiclass":
            config.audio_model_config.label_names = ['P', 'B', 'SR', 'ISR', 'MUR', 'no-stutter', 'other']
        elif config.audio_model_config.classification_task == "binaryclass":
            config.audio_model_config.label_names = ["any", "no-stutter"]
        else: raise ValueError("未定義のclassification taskが指定されています．")
        
        return FrameLevelAudioClassificationModule(config.audio_model_config)
    
    elif config.modality == "pretrain":
        if config.audio_model_config.classification_task == "multilabel":
            config.audio_model_config.label_names = ['SoundRep', 'WordRep', 'Prolongation', 'Block', 'Interjection', "any"]
            if config.audio_model_config.architecture.split("_")[0] == "ensamble":
                return Sep28kAudioClassificationModule(config.audio_model_config) # 一応
        elif config.audio_model_config.classification_task == "multiclass":
            config.audio_model_config.label_names = ["SoundRep", "WordRep", "Prolongation", "Block", "Interjection", "no-stutter", "other"]
        elif config.audio_model_config.classification_task == "binaryclass":
            config.audio_model_config.label_names = ["any", "no-stutter"]
        else: raise ValueError("未定義のclassification taskが指定されています")
        return Sep28kAudioClassificationModule(config.audio_model_config)
            
    elif config.modality == "video":
        config.video_model_config.label_names = ['FG', 'HM', 'V', 'any']
        return VideoClassificationModule(config.video_model_config)
    elif config.modality == "multimodal":
        model_config = MultimodalModelConfig(
            audio_config=config.audio_model_config,
            video_config=config.video_model_config,
            fusion_method=config.fusion_method,
            fusion_dim=config.fusion_dim
        )
        model_config.label_names = ['P', 'B', 'SR', 'ISR', 'MUR', 'FG', 'HM', 'V', 'any']
        return MultimodalClassificationModule(model_config)
    else:
        raise ValueError(f"Unsupported modality: {config.modality}")
    

if __name__ == "__main__":

    audio_config = AudioModelConfig()
    video_config = VideoModelConfig()
    multimodal_config = MultimodalModelConfig(audio_config=audio_config, video_config=video_config)
    
    training_config = TrainingConfig(
        modality="multimodal",
        audio_model_config=audio_config,
        video_model_config=video_config,
        fusion_method="concat",
        fusion_dim=512
    )


    model = prep_model(training_config)
    print(model)
    # dummy input for testing
    dummy_audio_input = {
        "audio_inputs": {
            "input_values": torch.randn(2, 16000),  # (batch_size, channels, length)
            "attention_mask": torch.ones(2, 16000)  # (batch_size, length)
        },
        "video_inputs": {
            "pixel_values": torch.randn(2, 32, 3, 224, 224)  # (batch_size, channels, height, width)
        }
    }

    outputs = model(**dummy_audio_input["audio_inputs"], **dummy_audio_input["video_inputs"])
    print(outputs["logits"].shape) 