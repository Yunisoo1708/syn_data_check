#!/bin/bash
cd /mnt/c/Research/RFD4_fine_tune
bash convert_batch.sh /mnt/c/Research/ncflow_ref/ncflow/out2_klone_bal /mnt/c/Research/RFD4_fine_tune/data/bal_pocket12a bal > /mnt/c/Research/RFD4_fine_tune/logs/data_generation/convert_bal_pocket12a.log 2>&1
