# 数据集使用说明
📢 `processed`文件为预处理过的数据集，其余为原数据集

1. GHG
    - `ghg.npy`的shape ==> (seq_len:327, num_devices:7, num_features:16)
    - `ghg_mask_xx.npy`的shape与上同，`xx`表示缺失率，该类数组为`0/1`数组，`1`表示`observed value`，`0`表示`missing value`
   
2. PRSA
    同上

          
    
    