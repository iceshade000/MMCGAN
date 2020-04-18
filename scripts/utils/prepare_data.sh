#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3,4,5 python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
CUDA_VISIBLE_DEVICES=2,3,4,5 python calculate_inception_moments.py --dataset I128_hdf5 --data_root data