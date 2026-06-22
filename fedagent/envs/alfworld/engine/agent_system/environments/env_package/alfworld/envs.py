import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

@ray.remote(num_cpus=0.2)
class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, worker_idx=None):
        self.worker_idx = worker_idx
        self.env = base_env.init_env(batch_size=1, worker_idx=worker_idx)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, env_kwargs={}, client_id=None, client_num=None, min_goals_per_client=100, val_batch_size=500, partition_strategy='uniform', start_idx=None, end_idx=None, **partition_kwargs):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        # Federated learning parameters
        self.client_id = client_id
        self.client_num = client_num
        self.min_goals_per_client = min_goals_per_client
        self.val_batch_size = val_batch_size
        self.partition_strategy = partition_strategy
        self.partition_kwargs = partition_kwargs  # Extra keyword arguments for the partition strategy
        self.is_train = is_train

        # Batch-inference parameters (index window into the dataset)
        self.start_idx = start_idx
        self.end_idx = end_idx

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        # Forward start_idx and end_idx into partition_kwargs so the partition strategy can use them
        if start_idx is not None and end_idx is not None:
            partition_kwargs['start_idx'] = start_idx
            partition_kwargs['end_idx'] = end_idx
            
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset, client_id=client_id, client_num=client_num, partition_strategy=partition_strategy, min_games_per_client=min_goals_per_client, **partition_kwargs)
        # breakpoint()
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        self.workers = []
        for i in range(self.num_processes):
            worker = AlfworldWorker.remote(config, seed + (i // self.group_n), base_env, worker_idx=i)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, env_kwargs={}, client_id=None, client_num=None, min_goals_per_client=100, val_batch_size=500, partition_strategy='uniform', start_idx=None, end_idx=None, **partition_kwargs):
    """
    Build a set of ALFWorld environments for training or evaluation.
    
    Args:
        alf_config_path: Path to ALFWorld configuration file
        seed: Random seed for environment initialization
        env_num: Number of environments
        group_n: Number of environments per group
        is_train: Whether this is for training (True) or validation (False)
        env_kwargs: Additional environment keyword arguments
        client_id: Client ID for federated learning (None for non-federated)
        client_num: Total number of clients for federated learning
        min_goals_per_client: Minimum number of goals per client for federated learning
        val_batch_size: Number of validation games to use
        partition_strategy: Data partitioning strategy ('uniform', 'preference', 'coverage')
        start_idx: Start index for batch inference (None for all samples)
        end_idx: End index for batch inference (None for all samples)
        **partition_kwargs: Additional parameters for partition strategies
    """
    return AlfworldEnvs(
        alf_config_path, 
        seed, 
        env_num, 
        group_n, 
        is_train, 
        env_kwargs, 
        client_id=client_id, 
        client_num=client_num,
        min_goals_per_client=min_goals_per_client,
        val_batch_size=val_batch_size,
        partition_strategy=partition_strategy,
        start_idx=start_idx,
        end_idx=end_idx,
        **partition_kwargs
    )