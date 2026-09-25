import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from models import BaseModule
from configs import MyAudioModelConfig
from transformers import WavLMModel

class FrameLevelAudioClassificationModule(BaseModule):
    """WavLM-large-based stuttering classification model]
    stuttering detectionで使用したモデル構造を再現"""

    def __init__(self, config: MyAudioModelConfig):
        super().__init__(config)
        
        # Load pre-trained WavLM-large model
        self.backbone = WavLMModel.from_pretrained(config.pretrained_model_name) # 1028
        
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
                nn.LSTM(hidden_size, (hidden_size // 2), bidirectional=True, num_layer=2, batch_first=True),
                nn.Linear(hidden_size, (hidden_size // 2)),
                nn.functional.relu(),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(config.dropout),
                nn.Linear((hidden_size // 2), 1)  # Binary classification for each label (T,1)
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
        
        # # Pool hidden states
        # if attention_mask is not None:
        #     # Masked mean pooling
        #     pooled = self._masked_mean(hidden_states, attention_mask)
        # else:
        #     # Global mean pooling
        #     pooled = hidden_states.mean(dim=1)  # (B, D)
        
        # Classification head
        logits = {}
        for label, head in self.classifier.items():
            logits[label] = head(hidden_states)
        # Convert logits to a single tensor
        logits = torch.cat([logits[label] for label in self.label_names], dim=1) # shape (B, T, num_labels)
        
        return {
            "logits": logits,
            "pooled_output": None,
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