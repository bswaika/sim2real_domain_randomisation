import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv
from AirSimEnv.airsim_lap_env import AirSimLapEnv as AirSimEnv

from vae_common import load_vae, make_encode_state_fn
from ppo import PPO
from reward_functions import reward_functions
from utils import compute_gae, VideoRecorder

from PIL import Image, ImageEnhance

hyper_params = {
    'model': {
        'ppo': {
            'learning_rate': 1e-4,
            'lr_decay': 1.0,
            'epsilon': 0.2,
            'initial_std': 1.0,
            'value_scale': 1.0,
            'entropy_scale': 0.01,
            'model_name': f'dr-model-{int(time.time())}'
        },
        'vae': {
            'model_name': 'seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data',
            'model_type': 'cnn',
            'z_dim': 64
        },
        'horizon': 128, #256,
        'epochs': 2, #20,
        'episodes': 5, #5000,
        'batch_size': 32,
        'gae_lambda': 0.95,
        'discount_factor': 0.99,
        'eval_interval': 2, #1000   
    },
    'env': {
        'common': {
            'host': '172.26.0.1',
            'fps': 18,
            'action_smoothing': 0.3,
            'reward_fn': 'reward_speed_centering_angle_multiply',
            'obs_res': (160, 80)
        },
        'source': {
            'synchronous': True,
            'start_carla': False
        },
        'target': {
            'route_file': './AirSimEnv/routes/dr-test-02.txt'
        }
    },
    'dr': {
        'brightness': {
            'mu': 6.0,
            'sigma': 2.0
        },
        'contrast': {
            'mu': 6.0,
            'sigma': 2.0
        },
        'hue': {
            'mu': 6.0,
            'sigma': 2.0
        },
        'epochs': 5,
        'learning_rate': 1e-2
    }
}

class DRParameters:
    def __init__(self, hyper_params=hyper_params['dr']):
        self.hyper_params = hyper_params
        with tf.variable_scope('brightness'):
            self.brightness = tfp.distributions.Normal(
                tf.Variable(self.hyper_params['brightness']['mu'], name='b_mu'),
                tf.Variable(self.hyper_params['brightness']['sigma'], name='b_sigma')
            )
        with tf.variable_scope('contrast'):
            self.contrast = tfp.distributions.Normal(
                tf.Variable(self.hyper_params['contrast']['mu'], name='c_mu'),
                tf.Variable(self.hyper_params['contrast']['sigma'], name='c_sigma'),
            )
        with tf.variable_scope('hue'):    
            self.hue = tfp.distributions.Normal(
                tf.Variable(self.hyper_params['hue']['mu'], name='h_mu'),
                tf.Variable(self.hyper_params['hue']['sigma'], name='h_sigma'),
            )
    
    def sample(self, session):
        return session.run([self.brightness.sample(), self.contrast.sample(), self.hue.sample()])

    def get_trainable_params(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

class DomainRandomizer:
    def __init__(self, hyper_params=hyper_params):
        self.hyper_params = hyper_params
        self.init_vae()
        self.measurements = set(['steer', 'throttle', 'speed'])
        self.encode_state_fn = make_encode_state_fn(self.measurements)
        self.init_source_env()
        self.init_target_env()
        self.action_space = self.source_env.action_space
        self.num_actions = self.action_space.shape[0]
        self.params = DRParameters(self.hyper_params['dr'])
        self.trainable_params = {
            'b': self.params.get_trainable_params('brightness'),
            'c': self.params.get_trainable_params('contrast'),
            'h': self.params.get_trainable_params('hue'),
        }
        self.best_eval_reward = -np.inf
        self.epochs = self.hyper_params['dr']['epochs']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hyper_params['dr']['learning_rate'], name='dr_optimizer')
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['b']))
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['c']))
        self.optimizer.apply_gradients(zip([0.0, 0.0], self.trainable_params['h']))
        self.init_ppo()

    def init_source_env(self):
        self.source_env = CarlaEnv(
            host=self.hyper_params['env']['common']['host'],
            obs_res=self.hyper_params['env']['common']['obs_res'],
            encode_state_fn=self.encode_state_fn,
            reward_fn=reward_functions[self.hyper_params['env']['common']['reward_fn']],
            action_smoothing=self.hyper_params['env']['common']['action_smoothing'],
            fps=self.hyper_params['env']['common']['fps'],
            synchronous=self.hyper_params['env']['source']['synchronous'],
            start_carla=self.hyper_params['env']['source']['start_carla']
        )
    
    def init_target_env(self):
        self.target_env = AirSimEnv(
            host=self.hyper_params['env']['common']['host'],
            obs_res=self.hyper_params['env']['common']['obs_res'],
            encode_state_fn=self.encode_state_fn,
            reward_fn=reward_functions[f"{self.hyper_params['env']['common']['reward_fn']}_airsim"],
            action_smoothing=self.hyper_params['env']['common']['action_smoothing'],
            fps=self.hyper_params['env']['common']['fps'],
            route_file=self.hyper_params['env']['target']['route_file']
        )

    def init_vae(self):
        self.vae = load_vae(
            os.path.join('./vae/models', self.hyper_params['model']['vae']['model_name']),
            self.hyper_params['model']['vae']['z_dim'], 
            self.hyper_params['model']['vae']['model_type']
        )
    
    def init_ppo(self):
        self.input_shape = np.array([self.vae.z_dim + len(self.measurements)])
        self.model = PPO(
            self.input_shape, self.action_space,
            learning_rate=self.hyper_params['model']['ppo']['learning_rate'],
            lr_decay=self.hyper_params['model']['ppo']['lr_decay'],
            epsilon=self.hyper_params['model']['ppo']['epsilon'],
            initial_std=self.hyper_params['model']['ppo']['initial_std'],
            value_scale=self.hyper_params['model']['ppo']['value_scale'],
            entropy_scale=self.hyper_params['model']['ppo']['entropy_scale'],
            model_dir=os.path.join('./models', self.hyper_params['model']['ppo']['model_name'])
        )
        
        self.model.init_session()
        self.model.load_latest_checkpoint()
        self.model.write_dict_to_summary('hyperparams/ppo', self.hyper_params['model']['ppo'], 0)
        self.model.write_dict_to_summary('hyperparams/vae', self.hyper_params['model']['vae'], 0)
        self.model.write_dict_to_summary('hyperparams/general', {k:self.hyper_params['model'][k] for k in self.hyper_params['model'] if k != 'ppo' and k != 'vae'}, 0)

    def transform_frame(self, frame, transform_params):
        frame = Image.fromarray(frame)
        brightness = ImageEnhance.Brightness(frame)
        contrast = ImageEnhance.Contrast(frame)
        hue = ImageEnhance.Color(frame)
        frame = brightness.enhance(transform_params[0])
        frame = contrast.enhance(transform_params[1])
        frame = hue.enhance(transform_params[2])
        frame = np.asarray(frame)
        return frame

    def normalize_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0
        return frame

    def make_state(self, state, for_source_env=True, transform_params=(0.0, 0.0, 0.0), save_frame_idx=-1):
        frame, measurements = state['frame'], state['measurements']
        if for_source_env:
            frame = self.transform_frame(frame, transform_params)
            if save_frame_idx > -1:
                Image.fromarray(frame).save(os.path.join(self.model.image_dir, f'epoch-{save_frame_idx}.png'))
        frame = self.normalize_frame(frame)            
        encoded_state = self.vae.encode([frame])[0]
        encoded_state = np.append(encoded_state, measurements)
        return encoded_state

    def train(self, idx, transform_params):
        self.model.reset_episode_idx()

        episodes = self.hyper_params['model']['episodes']
        epochs = self.hyper_params['model']['epochs']
        batch_size = self.hyper_params['model']['batch_size']
        horizon = self.hyper_params['model']['horizon']

        gae_lambda = self.hyper_params['model']['gae_lambda']
        discount_factor = self.hyper_params['model']['discount_factor']

        while episodes <= 0 or self.model.get_episode_idx() < episodes:
            episode_idx = self.model.get_episode_idx()

            state, terminal_state, total_reward = self.source_env.reset(), False, 0
            state = self.make_state(state, transform_params=transform_params)

            print(f"Episode {episode_idx} (Step {self.model.get_train_step_idx()})")
            while not terminal_state:
                states, taken_actions, values, rewards, dones = [], [], [], [], []
                for _ in range(horizon):
                    action, value = self.model.predict(state, write_to_summary=True)
                    next_state, reward, terminal_state, info = self.source_env.step(action)
                    next_state = self.make_state(next_state, transform_params=transform_params)
                    if info['closed']:
                        sys.exit(0)
                    self.source_env.extra_info.extend([
                        "Episode {}".format(episode_idx),
                        "Training...",
                        "",
                        "Value:  % 20.2f" % value
                    ])

                    self.source_env.render()
                    total_reward += reward

                    states.append(state)         # [T, *input_shape]
                    taken_actions.append(action) # [T,  num_actions]
                    values.append(value)         # [T]
                    rewards.append(reward)       # [T]
                    dones.append(terminal_state) # [T]
                    state = next_state

                    if terminal_state:
                        break
                
                _, last_values = self.model.predict(state)

                advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
                returns = advantages + values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                states        = np.array(states)
                taken_actions = np.array(taken_actions)
                returns       = np.array(returns)
                advantages    = np.array(advantages)

                T = len(rewards)
                assert states.shape == (T, *self.input_shape)
                assert taken_actions.shape == (T, self.num_actions)
                assert returns.shape == (T,)
                assert advantages.shape == (T,)

                self.model.update_old_policy()
                for _ in range(epochs):
                    num_samples = len(states)
                    indices = np.arange(num_samples)
                    np.random.shuffle(indices)
                    for i in range(int(np.ceil(num_samples / batch_size))):
                        begin = i * batch_size
                        end   = begin + batch_size
                        if end > num_samples:
                            end = None
                        mb_idx = indices[begin:end]

                        self.model.train(states[mb_idx], taken_actions[mb_idx],
                                    returns[mb_idx], advantages[mb_idx])
            
            self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/reward", total_reward, episode_idx)
            self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/distance_traveled", self.source_env.distance_traveled, episode_idx)
            self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/average_speed", 3.6 * self.source_env.speed_accum / self.source_env.step_count, episode_idx)
            self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/center_lane_deviation", self.source_env.center_lane_deviation, episode_idx)
            self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/average_center_lane_deviation", self.source_env.center_lane_deviation / self.source_env.step_count, episode_idx)
            self.model.write_value_to_summary(f"train/{idx}|{'|'.join(map(str, transform_params))}/distance_over_deviation", self.source_env.distance_traveled / self.source_env.center_lane_deviation, episode_idx)
            self.model.write_episodic_summaries()
    
    def eval(self, idx, in_source_env=True, transform_params=(0.0, 0.0, 0.0)):
        env = self.source_env if in_source_env else self.target_env
        if in_source_env:
            state, terminal, total_reward = env.reset(is_training=False), False, 0
            state = self.make_state(state, transform_params=transform_params, save_frame_idx=idx)
        else:
            state, terminal, total_reward = env.reset(), False, 0
            state = self.make_state(state, for_source_env=False)

        rendered_frame = env.render(mode='rgb_array')

        if not in_source_env:
            filename = os.path.join(self.model.video_dir, f"epoch-{idx}-drparams-{'-'.join(map(str, transform_params))}.avi")
            video_recorder = VideoRecorder(filename, frame_size=rendered_frame.shape, fps=env.fps)
            video_recorder.add_frame(rendered_frame)
        
        episode_idx = self.model.get_episode_idx()

        while not terminal:
            if in_source_env:
                env.extra_info.append("Episode {}".format(episode_idx))
                env.extra_info.append("Running eval...".format(episode_idx))
                env.extra_info.append("")
            
            action, _ = self.model.predict(state, greedy=True)
            state, reward, terminal, info = env.step(action)
            
            if in_source_env:
                state = self.make_state(state, transform_params=transform_params)
            else:
                state = self.make_state(state, for_source_env=False)
            
            if info['closed']:
                break

            rendered_frame = env.render(mode='rgb_array')
            if not in_source_env:
                video_recorder.add_frame(rendered_frame)
            total_reward += reward
        
        if not in_source_env:
            video_recorder.release()
        
        if info['closed']:
            sys.exit(0)

        if in_source_env:
            self.model.write_value_to_summary("dr_params/brightness", transform_params[0], idx)
            self.model.write_value_to_summary("dr_params/contrast", transform_params[1], idx)
            self.model.write_value_to_summary("dr_params/hue", transform_params[2], idx)
        else:
            self.model.write_value_to_summary("dr_params/brightness_mean", self.model.sess.run(self.params.brightness.loc), idx)
            self.model.write_value_to_summary("dr_params/brightness_std", self.model.sess.run(self.params.brightness.scale), idx)
            self.model.write_value_to_summary("dr_params/contrast_mean", self.model.sess.run(self.params.contrast.loc), idx)
            self.model.write_value_to_summary("dr_params/contrast_std", self.model.sess.run(self.params.contrast.scale), idx)
            self.model.write_value_to_summary("dr_params/hue_mean", self.model.sess.run(self.params.hue.loc), idx)
            self.model.write_value_to_summary("dr_params/hue_std", self.model.sess.run(self.params.hue.scale), idx)

        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/reward", total_reward, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/distance_traveled", env.distance_traveled, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/average_speed", 3.6 * env.speed_accum / env.step_count, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/center_lane_deviation", env.center_lane_deviation, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/average_center_lane_deviation", env.center_lane_deviation / env.step_count, idx)
        self.model.write_value_to_summary(f"eval/{'source' if in_source_env else 'target'}/distance_over_deviation", env.distance_traveled / env.center_lane_deviation, idx)
        
        return total_reward
    
    def run(self):
        # self.model.sess.run(tf.variables_initializer(self.optimizer.variables()))
        for idx in range(self.epochs):
            print(f'DR Epoch {idx}')
            transform_params = self.params.sample(self.model.sess)
            self.train(idx, transform_params)
            source_reward = self.eval(idx, transform_params=transform_params)
            target_reward = self.eval(idx, in_source_env=False, transform_params=transform_params)

            if target_reward > self.best_eval_reward:
                self.model.save()
                self.best_eval_reward = target_reward

            
            self.transfer_loss = (source_reward - target_reward)
            bl = -tf.math.log(self.params.brightness.prob(transform_params[0]) * self.transfer_loss)
            cl = -tf.math.log(self.params.contrast.prob(transform_params[1]) * self.transfer_loss)
            hl = -tf.math.log(self.params.hue.prob(transform_params[2]) * self.transfer_loss)
            
            bg = tf.gradients(bl, self.trainable_params['b'])
            cg = tf.gradients(cl, self.trainable_params['c'])
            hg = tf.gradients(hl, self.trainable_params['h'])

            self.model.sess.run([
                self.optimizer.apply_gradients(zip(bg, self.trainable_params['b'])),
                self.optimizer.apply_gradients(zip(cg, self.trainable_params['c'])),
                self.optimizer.apply_gradients(zip(hg, self.trainable_params['h'])),
            ])
        
        with open(os.path.join(self.model.model_dir, 'converged_params.txt'), 'w') as outfile:
            outfile.write(f'Brightness ~ Normal(mu={self.model.sess.run(self.params.brightness.loc)}, sigma={self.model.sess.run(self.params.brightness.scale)})\n')
            outfile.write(f'Contrast ~ Normal(mu={self.model.sess.run(self.params.contrast.loc)}, sigma={self.model.sess.run(self.params.contrast.scale)})\n')
            outfile.write(f'Hue ~ Normal(mu={self.model.sess.run(self.params.hue.loc)}, sigma={self.model.sess.run(self.params.hue.scale)})\n')
        
        print('Done...')

if __name__ == '__main__':
    dr = DomainRandomizer()
    try:
        dr.run()
    except Exception as e:
        dr.source_env.close()
        raise e