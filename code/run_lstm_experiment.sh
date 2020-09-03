#!/bin/bash
for i in 02055100 02143000 12143600 11381500 03500000 14306500
do
   /home/georgy/anaconda3/envs/kaliv2/bin/python ann_script.py --basin_id=$i --model_name=LSTM --gpu=0 --batch_size=256 --history=4320
done