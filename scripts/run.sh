#!/bin/bash

LOG_FILE="./output.log"
nohup python run.py --dataset_name vindr --data_dir /media/nvidia/DATA2/datasets/original/vindr/ --out_dir /media/nvidia/DATA2/datasets/prepared/vindr/1024_cropped/ --batch_size 50 --n_workers 8 > "$LOG_FILE" 2>&1 &
echo "Script started in the background"
echo "Logs are being saved to output.log"