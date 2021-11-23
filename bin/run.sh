#!/usr/bin/env sh

python -m src.preprocess.peanut

python -m src.preprocess.run 'train' 'base' --batch-size=16 --num-workers=0 --epoch=50 --step-size=5 --lr=3e-4 --use-gpu --gpu=0 --apex
python -m src.preprocess.run 'train' 'refinement' --batch-size=16 --num-workers=0 --epoch=10 --step-size=5 --lr=5e-3 --use-gpu --gpu=0 --apex
python -m src.preprocess.run 'train' 'cls' --batch-size=16 --num-workers=0 --epoch=10 --step-size=5 --lr=5e-3 --use-gpu --gpu=0 --apex

python -m src.preprocess.run 'test' 'base' --batch-size=16 --num-workers=0 --use-gpu --gpu=3 --apex --snapshot='./checkpoints/model_apex_ep39.pt'
python -m src.preprocess.run 'test' 'refinement' --batch-size=16 --num-workers=0 --use-gpu --gpu=3 --apex --snapshot='./checkpoints/refinement_model_apex_ep5.pt'

python -m src.preprocess.run 'train' 'cls' --batch-size=16 --num-workers=0 --epoch=20 --step-size=5 --lr=3e-4 --use-gpu --gpu=0 --apex

python -m src.preprocess.lby_run 'train' 'openpose' --batch-size=32 --num-workers=0 --use-gpu --gpu=0 --apex --lr=3e-8 --epoch=60 --step-size=10
python -m src.preprocess.old_run 'train' 'openpose' --batch-size=32 --num-workers=0 --use-gpu --gpu=0 --apex --lr=3e-8 --epoch=60 --step-size=10
python -m src.preprocess.run 'train' 'openpose' --batch-size=32 --num-workers=0 --use-gpu --gpu=0 --apex --lr=3e-5 --epoch=60 --step-size=20
python -m src.preprocess.run 'test' 'openpose' --batch-size=16 --num-workers=0 --use-gpu --gpu=3 --snapshot='./checkpoints/openpose_model_apex_ep0.pt'

python -m src.preprocess.run 'train' 'openpose' --batch-size=32 --num-workers=0 --use-gpu --gpu=2 --apex --lr=3e-5 --epoch=60 --step-size=20 --reinforcement
python -m src.preprocess.run 'test' 'openpose' --batch-size=32 --num-workers=0 --use-gpu --gpu=2 --apex --reinforcement --snapshot='./checkpoints/openpose_model_apex_ep2.pt'

python -m src.preprocess.run 'train' 'openpose' --batch-size=16 --num-workers=0

python -m src.preprocess.run 'view' 'base'
