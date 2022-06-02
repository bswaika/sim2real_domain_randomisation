import os
import sys
import time

import numpy as np
import cv2

import airsim
import pygame
from pygame.locals import *

def get_camera_pose(type, vehicle_pose):
    position = vehicle_pose.position
    if type == 'spectator':
        X, Y, Z = -6, 0, -2.5
        return airsim.Pose(airsim.Vector3r(position.x_val+X, position.y_val+Y, position.z_val+Z), airsim.to_quaternion(-0.15, 0, 0))
    if type == 'dashboard':
        X, Y, Z = 0, 0, -1
        return airsim.Pose(airsim.Vector3r(position.x_val+X, position.y_val+Y, position.z_val+Z), airsim.to_quaternion(0.01, 0, 0))

class AirSimLapEnv():
    def __init__(self, host='127.0.0.1', port=41451, save_to='route.txt',
                 viewer_res=(1280, 720), fps=30, action_smoothing=0.9):        
        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.image_res = {
            'spec': (width, height)
        }
        self.action_smoothing = action_smoothing
        self.save_path = save_to

        try:
            self.client = airsim.CarClient(host, port)
            self.client.confirmConnection()
            self.reset(spawn_at=(np.nan, np.nan, np.nan))
        except Exception as e:
            self.close()
            raise e

    def reset(self, spawn_at=(85.341, -18.396, 1.338), end_at=(24.602, 42.973, 1.345)):
        self.client.reset()
        self.client.armDisarm(True)
        self.client.enableApiControl(True)
        self.lap_start_pose = self.client.simGetVehiclePose()
        self.adjust_cams()

        spawn_pose = airsim.Pose(airsim.Vector3r(spawn_at[0], spawn_at[1], spawn_at[2]), airsim.to_quaternion(np.nan, np.nan, np.nan))
        self.client.simSetVehiclePose(spawn_pose, True)
        self.lap_start_pose = self.client.simGetVehiclePose()

        destination_pose = airsim.Pose(airsim.Vector3r(end_at[0], end_at[1], end_at[2]), airsim.to_quaternion(0, 0, 0))
        self.lap_end_pose = destination_pose

        self.viewer_image = None
        self.step_count = 0
        self.point_queue = []
        self.terminal_state = False

        self.controls = {
            'steer': 0.0,
            'throttle': 0.0
        }

        self.previous_position = self.lap_start_pose.position
        time.sleep(2)

        return self.step(None)

    def step(self, action):        
        self.clock.tick_busy_loop(self.fps)

        if action is not None:
            steer, throttle = [float(a) for a in action]

            controls = airsim.CarControls(
                throttle = throttle * (1 - self.action_smoothing) + self.controls['throttle'] * self.action_smoothing, 
                steering = steer * (1 - self.action_smoothing) + self.controls['steer'] * self.action_smoothing
            )
            self.client.setCarControls(controls)
            self.controls = {
                'steer': controls.steering,
                'throttle': controls.throttle
            }

        responses = self.client.simGetImages([self.speccam])
        images = self.retrieve_image_arrays(responses)
        self.viewer_image = images['spec']

        self.point_queue.append(f'{round(self.previous_position.x_val, 3)} {round(self.previous_position.y_val, 3)} {round(self.previous_position.z_val, 3)}\n')
        self.previous_position = self.client.simGetVehiclePose().position

        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.terminal_state = True
        if pygame.key.get_pressed()[K_SPACE]:
            try:
                self.save()
            except Exception as e:
                print(str(e))
                self.terminal_state = True
        
        return self.terminal_state

    def save(self):
        if 'routes' not in os.listdir():
            raise FileNotFoundError
        if self.save_path in os.listdir('./routes'):
            raise FileExistsError
        with open(f'./routes/{self.save_path}', 'w') as outfile:
            outfile.writelines(self.point_queue)             

    def adjust_cams(self):
        self.speccam = airsim.ImageRequest(4, airsim.ImageType.Scene, compress=False)
        self.speccam_pose = get_camera_pose('spectator', self.lap_start_pose)
        self.client.simSetCameraPose(4, self.speccam_pose)


    def retrieve_image_arrays(self, responses):
        images = {
            'spec': airsim.string_to_uint8_array(responses[0].image_data_uint8).reshape(responses[0].height, responses[0].width, 3)
        }
        images['spec'] = cv2.resize(images['spec'], dsize=self.image_res['spec'])
        images['spec'] = cv2.cvtColor(images['spec'], cv2.COLOR_BGR2RGB)
        return images
    
    def render(self, mode='human'):
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
        pygame.display.flip()
        return None
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = AirSimLapEnv(host=sys.argv[1], save_to=sys.argv[2], action_smoothing=0.75)
    action = [0.0, 0.0]
    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_LEFT] or keys[K_a]:
            action[0] = -0.5
        elif keys[K_RIGHT] or keys[K_d]:
            action[0] = 0.5
        else:
            action[0] = 0.0
        action[0] = np.clip(action[0], -1, 1)
        action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0

        done = env.step(action)
        env.render()
        if done: break
    env.close()