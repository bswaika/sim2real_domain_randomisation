import os, sys

if 'CARLA_PATH' not in os.environ:
    os.environ['CARLA_PATH'] = '/home/bswaika/Downloads/carla-9-5/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg'
    
sys.path.append(os.environ['CARLA_PATH'])