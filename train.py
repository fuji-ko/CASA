import sys
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from configs import from_args
from models import prep_model
from dataset import prep_dataset, collate_fn
from finetuning_callbacks import UnfreezeEncoderOnPlateau

"""
--command example--

nohup python train.py --modality exp_audio --dataset_root "/work/abelab5/k_fuji/CASA/data/Voices-AWS/wav2vec2/dataset_duration=3_overlap=0" --dataset_annotator "A2" --audio_feature_extractor "wavlm" --audio_pretrained_model_name 'microsoft/wavlm-large' --output_dir "/work/abelab5/k_fuji/CASA/exp/wavlm" > try_wavlm.log 2>&1 &

"""
"""
BooleanOptionalAction使用例
true時：--dataset_do_single_label
false時：--no-dataset_do_single_label

nohup python train.py --modality exp_audio  --dataset_root "/work/abelab5/k_fuji/CASA/data/Voices-AWS/whisper" --dataset_annotator "A2" --audio_feature_extractor "whisper" --audio_pretrained_model_name "openai/whisper-medium" --output_dir "/work/abelab5/k_fuji/CASA/exp/train_extractor/whisper" --audio_learning_rate 1e-5 --batch_size 4 --max_epochs 50 > try_whisper_lr=1.0e-5.log 2>&1 &
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Train a stuttering detection model")
    # dataset arguments
    parser.add_argument('--dataset_root', type=str, default='data/clips', help='Root directory for dataset')
    parser.add_argument('--dataset_annotator', type=str, default='bau', help='Path to the label file')
    parser.add_argument('--dataset_label', type=str, default=None, help='Label to use for upsampling')
    parser.add_argument('--dataset_sampling_rate', type=int, default=16000, help='Sampling rate for audio')
    parser.add_argument('--dataset_symptom_names', type=str, nargs="+", default=['P', 'B', 'SR', 'ISR', 'MUR'])
    parser.add_argument('--dataset_do_single_label', action=argparse.BooleanOptionalAction, default=False, help='if you load val and test loader, select single annotation label')
    parser.add_argument('--dataset_do_speaker_adaptation', action=argparse.BooleanOptionalAction, default=False, help="この設定を有効にした場合，テストデータの各話者クリップからランダムに10%づつ取り出し学習データに加える")
    
    # audio model arguments
    parser.add_argument('--audio_feature_extractor', type=str, default='wav2vec2', choices=['wav2vec2', 'wavlm', 'whisper'], help="feture extractor name")
    parser.add_argument('--audio_pretrained_model_name', type=str, default='facebook/wav2vec2-base-960h', choices=['facebook/wav2vec2-base-960h', 'microsoft/wavlm-large', 'openai/whisper-medium'], help='Pretrained audio model name')
    parser.add_argument('--audio_freeze_encoder', action=argparse.BooleanOptionalAction, default=True, help='Freeze encoder layers of audio model')
    parser.add_argument('--audio_freeze_feature_extractor', action=argparse.BooleanOptionalAction, default=True, help='Freeze feature extractor layers of audio model')
    parser.add_argument('--audio_unfreeze_layers', type=int, default=None, help='Unfreeze layers of audio model')
    # added argument
    parser.add_argument('--audio_learning_rate', type=float, default=5e-5, help='Learning rate for audio model')
    parser.add_argument('--audio_iou_threshold', type=float, default=0.5, help='intersection of union of threshold')
    parser.add_argument('--audio_do_median_filter', action=argparse.BooleanOptionalAction, default=True, help="when test stage is started, preds item is used median filter.")
    parser.add_argument('--audio_architecture', type=str, default="lstm_multihead", choices=["lstm_singlehead", "lstm_multihead", "ensamble_transformer", "ensamble_linear", "ensamble_lstm"], help="classifierに用いるアーキテクチャタイプを指定")
    parser.add_argument('--audio_classification_task', type=str, default="multilabel", choices=["multilabel", "multiclass", "binaryclass"], help="分類タスクの手法を指定")
    parser.add_argument('--audio_finetuned_model_path', type=str, default=None, help="fine-tuningしたextractorやclassifierを用いる場合，読み込み先のモデルパスを指定．")
    parser.add_argument("--audio_output_layer", type=str, default="lstm", choices=["lstm", "mid_linear", "last_linear"],help="ensambleモデル学習時に，前段モデルから引っ張ってくる層を指定.")

    # video model arguments
    parser.add_argument('--video_pretrained_model_name', type=str, default='google/vivit-b-16x2-kinetics400', help='Pretrained video model name')
    parser.add_argument('--video_freeze_backbone', action=argparse.BooleanOptionalAction, default=True, help='Freeze backbone layers of video model')
    parser.add_argument('--video_unfreeze_last_n_layers', type=int, default=2, help='Unfreeze last N layers of video model')
    parser.add_argument('--video_learning_rate', type=float, default=1e-5, help='Learning rate for video model')
    # multimodal model arguments
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention'], help='Fusion method for multimodal model')
    parser.add_argument('--fusion_dim', type=int, default=512, help='Fusion dimension for multimodal model')

    # training arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of epochs for training')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--modality', type=str, choices=['audio', 'video', 'multimodal', 'exp_audio', 'finetune'], required=True, help='Modality to train on. if you use model contained exp dirctory, you chose "exp_audio".')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for saving models and logs')
    parser.add_argument('--train_encoder', action=argparse.BooleanOptionalAction, default=False, help="if you train encoder after training classifier, do true")

    return parser.parse_args()



def main(args):
    config = from_args(args) # コマンドライン引数を階層構造を持ったconfigに振り分ける作業
    # pl.seed_everything(config.seed)
    print(config)
    
    # Initialize logger
    # logger = WandbLogger(name="stuttering-detection", project="stuttering-detection", log_model=True, config=config)
    logger = TensorBoardLogger(os.path.join(config.output_dir, 'tb_logs'), name="{}-stuttering-detection".format(config.modality))
    logger.log_hyperparams(vars(config))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, 'checkpoints'),
        filename='stuttering-{epoch:02d}-{val_f1:.2f}',
        monitor='val/epoch_f1',
        mode='max',
        save_top_k=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val/epoch_f1',
        patience=10,
        mode='max'
    )

    train_dataset = prep_dataset(config.dataset_config,
                                 classification_task = config.audio_model_config.classification_task,
                                 split='train', 
                                 modality=config.modality, 
                                 feature_extractor=config.audio_model_config.feature_extractor, 
                                 pretrained_model_name= config.audio_model_config.pretrained_model_name)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    val_dataset = prep_dataset(config.dataset_config,
                               classification_task=config.audio_model_config.classification_task,
                               split='val', 
                               modality=config.modality, 
                               feature_extractor=config.audio_model_config.feature_extractor,
                               pretrained_model_name=config.audio_model_config.pretrained_model_name
                               )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )

    test_dataset = prep_dataset(config.dataset_config,
                                classification_task=config.audio_model_config.classification_task,
                                split='test', 
                                modality=config.modality, 
                                feature_extractor=config.audio_model_config.feature_extractor, 
                                pretrained_model_name=config.audio_model_config.pretrained_model_name)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )

    model = prep_model(config)

    # select trainer
    if config.train_encoder == True:
        print("training extractor.")
        unfreeze_encoder = UnfreezeEncoderOnPlateau(patience=3, min_delta=1e-4)
        # Trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator='gpu',
            devices=1,
            logger=logger,
            callbacks=[unfreeze_encoder, checkpoint_callback, early_stopping],
            gradient_clip_val=1.0
        )
    else:
        # Trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator='gpu',
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping],
            gradient_clip_val=1.0
        )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model,
                 test_loader,
                 ckpt_path="best")
    
    # save exprience setup
    # output_path = Path(config.output_dir)
    # with open(output_path / "config.yaml", "w") as f:
    #     yaml.dump(config, f, sort_keys=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
