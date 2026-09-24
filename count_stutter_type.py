"""total_datasetに含まれる話者数と吃音区間内の症状数の個数を算出"""
import pandas as pd
import csv
import numpy as np
import os
import sys

csv_path = "/work/abelab5/k_fuji/CASA/data/Voices-AWS/total_dataset.csv"
df = pd.read_csv(csv_path)
df_gold = df[df["annotator"] == "Gold"]
print(len(df_gold))
gold_list = df_gold.values
print(np.unique(gold_list[:,0]))

cols = ["SR", "ISR", "MUR", "P", "B"]
counts = (df_gold[cols]
        .value_counts()
        .reset_index(name="count"))
print(counts)