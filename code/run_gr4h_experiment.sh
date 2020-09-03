#!/bin/bash
for i in 02055100 02143000 12143600 11381500 03500000 14306500
do
   /home/georgy/anaconda3/envs/kaliv2/bin/python gr4h_script.py --basin_id=$i
done
