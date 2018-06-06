#!/bin/bash

PADDLE_TRAINING_ROLE=PSERVER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=2 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python ../Paddle/benchmark/fluid/fluid_benchmark.py --model resnet --device GPU --update_method pserver --iterations=10000 &

sleep 15

CUDA_VISIBLE_DEVICES=0,1 PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=2 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python ../Paddle/benchmark/fluid/fluid_benchmark.py --model resnet --device GPU --update_method pserver --iterations=10000 --gpus 2 &

CUDA_VISIBLE_DEVICES=2,3 PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=2 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=1 python ../Paddle/benchmark/fluid/fluid_benchmark.py --model resnet --device GPU --update_method pserver --iterations=10000 --gpus 2 &

