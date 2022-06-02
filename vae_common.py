import numpy as np
import re
from CarlaEnv.wrappers import vector
from vae.models import ConvVAE, MlpVAE

from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv
from AirSimEnv.airsim_lap_env import AirSimLapEnv as AirSimEnv

def load_vae(model_dir, z_dim=None, model_type=None):
    """
        Loads and returns a pretrained VAE
    """
    
    # Parse z_dim and model_type from name if None
    if z_dim is None: z_dim = int(re.findall("zdim(\d+)", model_dir)[0])
    if model_type is None: model_type = "mlp" if "mlp" in model_dir else "cnn"
    VAEClass = MlpVAE if model_type == "mlp" else ConvVAE
    target_depth = 1 if "seg_" in model_dir else 3

    # Load pre-trained variational autoencoder
    vae_source_shape = np.array([80, 160, 3])
    vae = VAEClass(source_shape=vae_source_shape,
                   target_shape=np.array([80, 160, target_depth]),
                   z_dim=z_dim, models_dir="vae",
                   model_dir=model_dir,
                   training=False)
    vae.init_session(init_logging=False)
    if not vae.load_latest_checkpoint():
        raise Exception("Failed to load VAE")
    return vae

def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame

def create_encode_state_fn(vae, measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    # Turn into bool array for performance
    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "orientation" in measurements_to_include]

    def encode_state(env):
        # Encode image with VAE
        frame = preprocess_frame(env.observation)
        encoded_state = vae.encode([frame])[0]
        
        # Append measurements
        measurements = []

        if isinstance(env, CarlaEnv):

            if measure_flags[0]: measurements.append(env.vehicle.control.steer)
            if measure_flags[1]: measurements.append(env.vehicle.control.throttle)
            if measure_flags[2]: measurements.append(env.vehicle.get_speed())

            # Orientation could be usedful for predicting movements that occur due to gravity
            if measure_flags[3]: measurements.extend(vector(env.vehicle.get_forward_vector()))
        
        if isinstance(env, AirSimEnv):
            if measure_flags[0]: measurements.append(env.controls['steer'])
            if measure_flags[1]: measurements.append(env.controls['throttle'])
            if measure_flags[2]: measurements.append(env.previous_state.speed)

            fwd = env.previous_state.kinematics_estimated.linear_velocity

            # Orientation could be usedful for predicting movements that occur due to gravity
            if measure_flags[3]: measurements.extend(np.array([fwd.x_val, fwd.y_val, fwd.z_val]))

        encoded_state = np.append(encoded_state, measurements)
        
        return encoded_state
    return encode_state

def make_encode_state_fn(measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    # Turn into bool array for performance
    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "orientation" in measurements_to_include]

    def encode_state(env):
        # Encode image with VAE
        frame = env.observation
        # encoded_state = vae.encode([frame])[0]
        
        # Append measurements
        measurements = []

        if isinstance(env, CarlaEnv):

            if measure_flags[0]: measurements.append(env.vehicle.control.steer)
            if measure_flags[1]: measurements.append(env.vehicle.control.throttle)
            if measure_flags[2]: measurements.append(env.vehicle.get_speed())

            # Orientation could be usedful for predicting movements that occur due to gravity
            if measure_flags[3]: measurements.extend(vector(env.vehicle.get_forward_vector()))
        
        if isinstance(env, AirSimEnv):
            if measure_flags[0]: measurements.append(env.controls['steer'])
            if measure_flags[1]: measurements.append(env.controls['throttle'])
            if measure_flags[2]: measurements.append(env.previous_state.speed)

            fwd = env.previous_state.kinematics_estimated.linear_velocity

            # Orientation could be usedful for predicting movements that occur due to gravity
            if measure_flags[3]: measurements.extend(np.array([fwd.x_val, fwd.y_val, fwd.z_val]))

        # encoded_state = np.append(encoded_state, measurements)
        
        return {
            'frame': frame,
            'measurements': measurements
        }
    return encode_state

# def create_encode_state_fn_airsim(vae, measurements_to_include):
#     """
#         Returns a function that encodes the current state of
#         the environment into some feature vector.
#     """

#     # Turn into bool array for performance
#     measure_flags = ["steer" in measurements_to_include,
#                      "throttle" in measurements_to_include,
#                      "speed" in measurements_to_include,
#                      "orientation" in measurements_to_include]

#     def encode_state(env):
#         # Encode image with VAE
#         frame = preprocess_frame(env.observation)
#         encoded_state = vae.encode([frame])[0]
        
#         # Append measurements
#         measurements = []
#         if measure_flags[0]: measurements.append(env.controls['steer'])
#         if measure_flags[1]: measurements.append(env.controls['throttle'])
#         if measure_flags[2]: measurements.append(env.previous_state.speed)

#         fwd = env.previous_state.kinematics_estimated.linear_velocity

#         # Orientation could be usedful for predicting movements that occur due to gravity
#         if measure_flags[3]: measurements.extend(np.array([fwd.x_val, fwd.y_val, fwd.z_val]))

#         encoded_state = np.append(encoded_state, measurements)
        
#         return encoded_state
#     return encode_state

