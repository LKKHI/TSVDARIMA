# sprasity 0.3-0.9
import os

for sprasity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]:
    print(f'start sprasity {sprasity}...')
    cmd = f'python main_sprasity.py --sprasity {sprasity}'
    os.system(cmd)