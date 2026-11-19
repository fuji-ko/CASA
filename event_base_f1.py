import sys
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple, Any

def _convert_to_segment(frames: np.ndarray) -> List[Tuple[int, int]]: # frame: (B, T)
    """frame列からsegment(吃音開始フレーム，吃音終了フレーム)に変換"""
    segments = []
    start = None

    for i, frame in enumerate(frames):
        if frame == 1 and start is None:
            start = i

        elif frame == 0 and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(frames)))

    return segments

def median_filter_1d(x, k=9):
    """
    x: 1d binary sequence(0/1)
    k: window size (odd(奇数) recommended)
    """
    assert k % 2 == 1
    
    pad = k // 2
    # padding関数
    # edge: 両端にある数値を延長させる
    # pad_width: 延長させる数値の個数
    x_pad = np.pad(x, pad_width=pad, mode="edge")
    out = np.zeros_like(x) # xと同じ次元の0行列を作成
    for i in range(len(x)):
        window = x_pad[i:i+k]
        out[i] = np.median(window)
    
    return out

def make_filter_list(preds: torch.Tensor,
                    labels:torch.Tensor,
                    label_names: List[str],
                    iou_threshold: float) -> Dict[str, float]:
    """
    input: (B,T,C)
    output: {"sr":, "wr":, "bl":, "p":, "b":}
    """
    window_list = range(3,21,2) # [3,5,7,...,19]
    window_dict = {} 
    f1_dict = {}
    preds = preds.cpu().numpy() # (total_batch, T, C)
    labels = labels.cpu().numpy()
    raw_preds = preds.copy()
    f1_dict = {}
    B,T,C = preds.shape
    
    if len(label_names) != C:
        raise ValueError(f"num dim is not same: {len(label_names)=}, {C=}")
    
    for label in label_names:
        f1_dict[label] = {} # window_lenとそれに対応するevent_f1の値が格納{"3":{"sr":{"tp"~,...,"f1":~}}, "5":~, ...}
        
    # activate mad filter
    for window_len in window_list:
        preds_med = raw_preds.copy()
        
        for b in range(B):
            for c in range(C):
                    preds_med[b, :, c] = median_filter_1d(raw_preds[b, :, c], window_len)
                    
        score_dict = event_based_f1(preds=preds_med, 
                                    labels=labels,
                                    label_names=label_names,
                                    iou_threshold=iou_threshold)
        
        for label in label_names:
            f1_dict[label][window_len] = score_dict[label]["f1"]
        
    # select best window length
    for label in label_names:
        max_f1_window_len = max(f1_dict[label], key=f1_dict[label].get) # f1_dict.get(=f1)の最大値のキー(k)を選択
        max_f1 = max(f1_dict[label].values())
        window_dict[label] = max_f1_window_len
        f1_dict[label] = max_f1
    
    return window_dict

def _event_iou(event_a: Tuple[int, int],
                event_b: Tuple[int, int],
                ) -> float:
    """二つのイベントのiouを計算する"""
    a_start, a_end = event_a
    b_start, b_end = event_b

    intersection_start = max(a_start, b_start)
    intersection_end   = min(a_end, b_end)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(a_start, b_start)
    union_end   = max(a_end, b_end)

    union = union_end - union_start

    if union == 0:
        return 0.0

    return intersection / union

def _match_events(gt_events: List[Tuple[int, int]],
                pred_events: List[Tuple[int, int]],
                iou_threshold: float = 0.5
                ) -> Tuple[int, int, int]:
    """GTeventとpred eventをIoUに基づいて1対1対応させる"""

    if len(gt_events) == 0:
        return 0, len(pred_events), 0

    if len(pred_events) == 0:
        return 0, 0, len(gt_events)

    iou_matrix = np.zeros(
        (len(gt_events), len(pred_events)),
        dtype=np.float32
    )

    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(pred_events):
            iou_matrix[gt_idx, pred_idx] = _event_iou(gt_event, pred_event)

    # IoUの大きい順にcandidate pairをならべる
    candidate_pairs = []

    for gt_idx in range(len(gt_events)):
        for pred_idx in range(len(pred_events)):
            iou = iou_matrix[gt_idx, pred_idx]

            if iou >= iou_threshold:
                candidate_pairs.append((iou, gt_idx, pred_idx))

    # 一つの正解区間に複数の予測区間が重なっていた場合，一番最初の区間を採用,それ以降はfalseで計算
    candidate_pairs.sort(key=lambda x: x[0],
                            reverse=True)

    # 1-to-1 matching
    # 貪欲法
    # 一つの正解区間に複数の予測区間が重なっていた場合，一番最初の区間を採用,それ以降はfalseで計算
    matched_gt = set()
    matched_pred = set()
    tp = 0
    for iou, gt_idx, pred_idx in candidate_pairs:
        # すでにmatching済みならskip
        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        
        tp += 1
    
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    
    return tp, fp, fn

def _calc_f1(tp: int,
                fp: int,
                fn: int
                ) -> Tuple[float, float, float]:

    precision = (tp / (tp + fp) if tp + fp > 0 else 0.0)
    recall = (tp/ (tp + fn) if tp + fn > 0 else 0.0)
    
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else: 
        f1 = 0.0
        
    return precision, recall, f1

def event_based_f1(preds: torch.Tensor | np.ndarray,
                    labels: torch.Tensor | np.ndarray,
                    label_names: List[str],
                    iou_threshold: float = 0.5
                    ) -> Dict[str, Dict[str, float]]:
    """event_based_scoreを計算"""
    
    # もしpredsとlabelの型がtensorだったら，numpy配列に変換
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    B,T,C = preds.shape
    results = {}
    
    for c, label_name in enumerate(label_names):
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for b in range(B):
            gt_sequence = labels[b, :, c]
            pred_sequence = preds[b, :, c]

            gt_events = _convert_to_segment(gt_sequence)
            pred_events = _convert_to_segment(pred_sequence)

            tp,fp,fn = _match_events(gt_events=gt_events,
                                    pred_events=pred_events,
                                    iou_threshold=iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        precision, recall, f1 = _calc_f1(tp=total_tp,
                                                fp=total_fp,
                                                fn=total_fn)
        
        results[label_name] = {
                                    "tp": total_tp,
                                    "fp": total_fp,
                                    "fn": total_fn,
                                    "precision": precision,
                                    "recall": recall,
                                    "f1": f1,
                                }
    return results