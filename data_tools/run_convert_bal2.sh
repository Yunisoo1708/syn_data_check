#!/bin/bash
cd /mnt/c/Research/RFD4_fine_tune
bash convert_batch.sh /mnt/c/Research/ncflow_ref/ncflow/out2_new_bal2 /mnt/c/Research/RFD4_fine_tune/data/bal_pocket12b bl2 > /mnt/c/Research/RFD4_fine_tune/logs/data_generation/convert_bal_pocket12b.log 2>&1
