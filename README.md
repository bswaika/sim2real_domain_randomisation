# Sim2Real Domain Randomisation for Autonomous Driving

<!---THIS DOCUMENT IS A WORK-IN-PROGRESS--->

## About the project
This project is based directly on the works of [Marcus Loo Vergara](https://github.com/bitsauce) who provides a great starting point for 
driving sim RL using carla in this [repo here](https://github.com/bitsauce/Carla-ppo). This project
extends his work for the purposes of Domain Randomisation to hopefully gain better sim2real transfer.
We have added a AirSim Gym Environment, and a DR pipeline which can be customized for any gym like 
environment. The DR pipeline is modular and can be modified to incorporate other gym environments,
and other agents. However, swapping the agent would mean having to rewrite the training loop.

## Stack
1. [Carla Simulator 0.9.5](https://github.com/carla-simulator/carla/releases/tag/0.9.5)
2. [AirSim 1.7 CityEnviron(Windows)](https://github.com/microsoft/AirSim/releases/tag/v1.7.0-windows)
3. [AirSim 1.7 AirSimNH(Linux)](https://github.com/microsoft/AirSim/releases/tag/v1.7.0-linux)
4. [Tensorflow 1.15](https://github.com/tensorflow/docs/tree/master/site/en/r1)
