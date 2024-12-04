#!/bin/bash


DATA1=human
DATA2=biosnap
DATA3=bindingdb
SPLIT=cold
#python main2.py --data  ${DATA1}
#python main2.py --data ${DATA2}
python main2.py --data ${DATA2} --split ${SPLIT}
python main2.py --data ${DATA3} --split ${SPLIT}