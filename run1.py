import torch
from data_provider.data_factory import data_provider
from exp.exp_text import parser
import argparse



args = parser.parse_args()

print("Args in experiment:")
print(args)

train_data , train_loader = data_provider(args, 'train')

#print(train_data.features)

for batch_idx, data in enumerate(train_loader):
    print(f"Batch {batch_idx}: {len(data)} elements")
    print(f"Data structure: {len(data[0][0]),len(data[1][0]),len(data[2][0]),len(data[3][0]) }")
    break  # 查看第一个批次的数据


