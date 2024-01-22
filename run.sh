#!/bin/sh		

f=logName  # modify as needed

python -u -m torch.distributed.run --master_port=3993 main.py --target "clipart" --devices 0 --file $f --epochs 40 --alpha 0.1 --beta .2 --seed 7777 
python -u -m torch.distributed.run --master_port=3993 main.py --target "infograph" --devices 0 --file $f --epochs 40 --alpha 0.1 --beta 1. --seed 7777 
python -u -m torch.distributed.run --master_port=3993 main.py --target "painting" --devices 0 --file $f --epochs 40 --alpha 0.1 --beta 1. --seed 7777 
python -u -m torch.distributed.run --master_port=3993 main.py --target "real" --devices 0 --file $f --epochs 40 --alpha 0.1 --beta 1. --seed 7777 
python -u -m torch.distributed.run --master_port=3993 main.py --target "quickdraw" --devices 0 --file $f --epochs 200 --alpha 0.1 --beta 1. --seed 7777  --lr 0.05
python -u -m torch.distributed.run --master_port=3993 main.py --target "sketch" --devices 0 --file $f --epochs 40 --alpha 0.1 --beta .2 --seed 7777 

