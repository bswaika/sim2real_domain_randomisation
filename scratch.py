# hyper_params = {
#     'model': {
#         'ppo': {
#             'learning_rate': 1e-4,
#             'lr_decay': 1.0,
#             'epsilon': 0.2,
#             'initial_std': 1.0,
#             'value_scale': 1.0,
#             'entropy_scale': 0.01,
#             'model_name': f'dr-model-{int(time.time())}'
#         },
#         'vae': {
#             'model_name': 'seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data',
#             'model_type': 'cnn',
#             'z_dim': 64
#         },
#         'horizon': 128, #256,
#         'epochs': 2, #20,
#         'episodes': 5, #5000,
#         'batch_size': 32,
#         'gae_lambda': 0.95,
#         'discount_factor': 0.99,
#         'eval_interval': 2, #1000   
#     },
#     'env': {
#         'common': {
#             'host': '172.26.0.1',
#             'fps': 18,
#             'action_smoothing': 0.3,
#             'reward_fn': 'reward_speed_centering_angle_multiply',
#             'obs_res': (160, 80)
#         },
#         'source': {
#             'synchronous': True,
#             'start_carla': False
#         },
#         'target': {
#             'route_file': './AirSimEnv/routes/dr-test-02.txt'
#         }
#     },
#     'dr': {
#         'brightness': {
#             'mu': 6.0,
#             'sigma': 2.0
#         },
#         'contrast': {
#             'mu': 6.0,
#             'sigma': 2.0
#         },
#         'hue': {
#             'mu': 6.0,
#             'sigma': 2.0
#         },
#         'epochs': 5,
#         'learning_rate': 1e-2
#     }
# }