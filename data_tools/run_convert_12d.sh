#!/bin/bash
cd /mnt/c/Research/RFD4_fine_tune
bash convert_batch.sh /mnt/c/Research/ncflow_ref/ncflow/out2_new_top /mnt/c/Research/RFD4_fine_tune/data/top_pocket12d tp4 > /mnt/c/Research/RFD4_fine_tune/logs/data_generation/convert_top_pocket12d.log 2>&1
