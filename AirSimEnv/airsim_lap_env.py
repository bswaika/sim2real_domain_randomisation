import sys
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
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

class AirSimLapEnv(gym.Env):

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host='127.0.0.1', port=41451,
                 viewer_res=(1280, 720), obs_res=(1280, 720),
                 reward_fn=None, encode_state_fn=None,
                 route_file='route.txt', fps=30, action_smoothing=0.9, trace=False):        
        pygame.init()
        pygame.font.init()

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        self.image_res = {
            'spec': (width, height),
            'dash': (out_width, out_height)
        }

        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.route_file = route_file
        self.trace = trace

        try:
            self.client = airsim.CarClient(host, port)
            self.client.confirmConnection()
            self.reset()
        except Exception as e:
            self.close()
            raise e

    def generate_route_waypoints(self):
        self.waypoints = []
        self.waypoints_list = []
        i, distance, gap = 0, 3, 2
        while i < len(self.route):
            j = i + 1
            while j < len(self.route) and self.route[j].distance_to(self.route[i]) < distance:
                j += 1
            if j >= len(self.route):
                break
            self.waypoints.append((self.route[i], self.route[j]))
            self.waypoints_list.append(self.route[i])
            self.waypoints_list.append(self.route[j])
            i = j + 1
            while i < len(self.route) and self.route[i].distance_to(self.route[j]) < gap:
                i += 1

    def load_route(self):
        filename = self.route_file if '/' in self.route_file else f'./routes/{self.route_file}'
        with open(filename, 'r') as infile:
            points = [line.split(' ') for line in infile.readlines()]
            points = [list(map(float, point)) for point in points]
            self.route = [airsim.Vector3r(point[0], point[1], point[2]) for point in points]
    
    def is_route_complete(self, position: airsim.Vector3r):
        DISTANCE_THRESHOLD = 1
        return position.distance_to(self.route[-1]) <= DISTANCE_THRESHOLD

    def get_closest_waypoints(self, position):
        total_distance = lambda p, p1, p2: np.linalg.norm(p-p1) + np.linalg.norm(p-p2)
        to_np_array = lambda vec: np.array([vec.x_val, vec.y_val, vec.z_val])

        car_pos = to_np_array(position)

        min_dist = np.inf
        min_i = 0
        for i in range(len(self.waypoints_list)-1):
            dist = total_distance(car_pos, to_np_array(self.waypoints_list[i]), to_np_array(self.waypoints_list[i+1]))
            if dist < min_dist:
                min_dist = dist
                min_i = i
        
        return to_np_array(self.waypoints_list[min_i]), to_np_array(self.waypoints_list[min_i+1])


    def reset(self, spawn_at=(85.341, -18.396, 1.338), end_at=(24.602, 42.973, 1.345)):
        self.client.reset()
        self.client.armDisarm(True)
        self.client.enableApiControl(True)
        self.lap_start_pose = self.client.simGetVehiclePose()
        self.adjust_cams()

        self.load_route()
        self.generate_route_waypoints()

        spawn_pose = airsim.Pose(self.route[0], airsim.to_quaternion(np.nan, np.nan, np.nan))
        self.client.simSetVehiclePose(spawn_pose, True)
        self.lap_start_pose = self.client.simGetVehiclePose()
        
        if self.trace:
            self.draw_waypoints()
        destination_pose = airsim.Pose(self.route[-1], airsim.to_quaternion(0, 0, 0))
        self.lap_end_pose = destination_pose

        self.terminal_state = False 
        self.closed = False 
        self.observation = None
        self.viewer_image = None
        self.start_t = time.time()
        self.step_count = 0

        self.total_reward = 0.0
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.laps_completed = 0.0

        self.controls = {
            'steer': 0.0,
            'throttle': 0.0
        }

        self.previous_position = self.lap_start_pose.position
        self.previous_state = self.client.getCarState()
        self.has_collided = self.client.simGetCollisionInfo().has_collided

        return self.step(None)[0]

    def draw_waypoints(self):
        points_start, points_end = [], []
        for start, end in self.waypoints:
            points_start.append(start)
            points_end.append(end)
        self.client.simPlotArrows(points_start, points_end, is_persistent=True)

    def remove_unwanted_objects(self):
        unwanted_regex_list = ['.*coupe.*', '.*sedan.*', '.*suv.*', '.*saloon.*', '.*hatchback.*', 'Vehicle.*']
        for regex in unwanted_regex_list:
            objects = self.client.simListSceneObjects(regex)
            for object in objects:
                self.client.simDestroyObject(object)

    def step(self, action):
        if self.closed:
            raise Exception('AirSimEnv.step() called after it was closed...')
        
        self.clock.tick_busy_loop(self.fps)

        if action is not None:
            self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5
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

        responses = self.client.simGetImages([self.dashcam, self.speccam])
        images = self.retrieve_image_arrays(responses)
        self.observation = images['dash']
        self.viewer_image = images['spec']

        position = self.client.simGetVehiclePose().position
        state = self.client.getCarState()
        has_collided = self.client.simGetCollisionInfo().has_collided

        step_distance = position.distance_to(self.previous_position)
        self.distance_traveled += step_distance if step_distance > 0.009 else 0
        self.speed_accum += state.speed

        wpa, wpb = self.get_closest_waypoints(position)
        position_as_array = np.array([position.x_val, position.y_val, position.z_val])
        self.distance_from_center = np.linalg.norm(position_as_array-wpb) if np.isclose(np.linalg.norm(wpa-wpb), 0) else (np.linalg.norm(np.cross(wpa-wpb, wpb-position_as_array)) / np.linalg.norm(wpa-wpb)) 
        self.center_lane_deviation += self.distance_from_center

        if self.is_route_complete(position):
            self.terminal_state = True

        self.previous_position = position
        self.previous_state = state
        self.has_collided = has_collided
        encoded_state = self.encode_state_fn(self)

        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            # print('here')
            self.terminal_state = True
        
        return encoded_state, self.last_reward, self.terminal_state, { 'closed': self.closed }

    def adjust_cams(self):
        self.dashcam = airsim.ImageRequest(0, airsim.ImageType.Scene, compress=False)
        self.speccam = airsim.ImageRequest(4, airsim.ImageType.Scene, compress=False)

        self.dashcam_pose = get_camera_pose('dashboard', self.lap_start_pose)
        self.speccam_pose = get_camera_pose('spectator', self.lap_start_pose)
        self.client.simSetCameraPose(0, self.dashcam_pose)
        self.client.simSetCameraPose(4, self.speccam_pose)


    def retrieve_image_arrays(self, responses):
        images = {
            'dash': airsim.string_to_uint8_array(responses[0].image_data_uint8).reshape(responses[0].height, responses[0].width, 3),
            'spec': airsim.string_to_uint8_array(responses[1].image_data_uint8).reshape(responses[1].height, responses[1].width, 3)
        }
        images['dash'] = cv2.resize(images['dash'], dsize=self.image_res['dash'])
        images['spec'] = cv2.resize(images['spec'], dsize=self.image_res['spec'])
        images['dash'] = cv2.cvtColor(images['dash'], cv2.COLOR_BGR2RGB)
        images['spec'] = cv2.cvtColor(images['spec'], cv2.COLOR_BGR2RGB)
        return images
    
    def render(self, mode='human'):
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)

        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation
    
    def close(self):
        pygame.quit()

def reward_fn(env: AirSimLapEnv):
    to_np_array = lambda vec: np.array([vec.x_val, vec.y_val, vec.z_val])
    fwd    = to_np_array(env.previous_state.kinematics_estimated.linear_velocity)
    # print(fwd)
    wpa, wpb = env.get_closest_waypoints(env.previous_position)
    wp_fwd = wpb - wpa
    if np.dot(fwd[:2], wp_fwd[:2]) > 0:
        return env.previous_state.speed
    return 0

if __name__ == '__main__':
    env = AirSimLapEnv(host=sys.argv[1], obs_res=(160, 80), route_file=sys.argv[2], reward_fn=reward_fn, trace=True)
    action = np.zeros(env.action_space.shape[0])
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

        obs, reward, done, info = env.step(action)
        # print(round(reward, 2), round(env.distance_traveled, 2), round(env.center_lane_deviation, 2))
        
        if info["closed"]:
            env.close()
            sys.exit(0)
        if done: break
        env.render()
    print(round(env.total_reward, 2))
    env.close()