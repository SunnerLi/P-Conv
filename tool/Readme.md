# Tool

Usage
---
```
$ python3 generate_mask.py --ratio_min 0.01 --ratio_max 0.1 --margin False --num 1000
$ python3 generate_mask.py --ratio_min 0.1 --ratio_max 0.2 --margin False --num 1000
$ python3 generate_mask.py --ratio_min 0.2 --ratio_max 0.3 --margin False --num 1000
$ python3 generate_mask.py --ratio_min 0.3 --ratio_max 0.4 --margin False --num 1000
$ python3 generate_mask.py --ratio_min 0.4 --ratio_max 0.5 --margin False --num 1000
$ python3 generate_mask.py --ratio_min 0.5 --ratio_max 0.6 --margin False --num 1000
$ python3 generate_mask.py --ratio_min 0.01 --ratio_max 0.1 --margin True --num 1000
$ python3 generate_mask.py --ratio_min 0.1 --ratio_max 0.2 --margin True --num 1000
$ python3 generate_mask.py --ratio_min 0.2 --ratio_max 0.3 --margin True --num 1000
$ python3 generate_mask.py --ratio_min 0.3 --ratio_max 0.4 --margin True --num 1000
$ python3 generate_mask.py --ratio_min 0.4 --ratio_max 0.5 --margin True --num 1000
$ python3 generate_mask.py --ratio_min 0.5 --ratio_max 0.6 --margin True --num 1000
```

or generate in one step:
```
$ python3 generate_mask.py --ratio_min 0.01 --ratio_max 0.1 --margin False --num 10 \
&& python3 generate_mask.py --ratio_min 0.1 --ratio_max 0.2 --margin False --num 10 \
&& python3 generate_mask.py --ratio_min 0.2 --ratio_max 0.3 --margin False --num 10 \
&& python3 generate_mask.py --ratio_min 0.3 --ratio_max 0.4 --margin False --num 10 \
&& python3 generate_mask.py --ratio_min 0.4 --ratio_max 0.5 --margin False --num 10 \
&& python3 generate_mask.py --ratio_min 0.5 --ratio_max 0.6 --margin False --num 10 \
&& python3 generate_mask.py --ratio_min 0.01 --ratio_max 0.1 --margin True --num 10 \
&& python3 generate_mask.py --ratio_min 0.1 --ratio_max 0.2 --margin True --num 10 \
&& python3 generate_mask.py --ratio_min 0.2 --ratio_max 0.3 --margin True --num 10 \
&& python3 generate_mask.py --ratio_min 0.3 --ratio_max 0.4 --margin True --num 10 \
&& python3 generate_mask.py --ratio_min 0.4 --ratio_max 0.5 --margin True --num 10 \
&& python3 generate_mask.py --ratio_min 0.5 --ratio_max 0.6 --margin True --num 10
```