import os
import json
import random

from tqdm import tqdm
from termcolor import colored

import textworld
import textworld.agents
import textworld.gym

from alfworld.agents.utils.misc import Demangler, add_task_to_grammar
from alfworld.agents.expert import HandCodedTWAgent, HandCodedAgentTimeout
from agent_system.environments.partition_strategy import partition_dataset, get_partition_info, visualize_alfworld_client_category_distribution, visualize_all_clients_category_distribution, visualize_coverage_normal_distribution


TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


class AlfredDemangler(textworld.core.Wrapper):

    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle = shuffle

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._entity_infos, shuffle=self.shuffle)
        for info in self._entity_infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


class AlfredInfos(textworld.core.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamefile = None

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._gamefile = args[0]

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        state["extra.gamefile"] = self._gamefile
        return state


# Enum for the supported types of AlfredExpert.
class AlfredExpertType:
    HANDCODED = "handcoded"
    PLANNER = "planner"


class AlfredExpert(textworld.core.Wrapper):

    def __init__(self, env=None, expert_type=AlfredExpertType.HANDCODED):
        super().__init__(env=env)

        self.expert_type = expert_type
        self.prev_command = ""
        if expert_type not in (AlfredExpertType.HANDCODED, AlfredExpertType.PLANNER):
            msg = "Unknown type of AlfredExpert: {}.\nExpecting either '{}' or '{}'."
            msg = msg.format(expert_type, AlfredExpertType.HANDCODED, AlfredExpertType.PLANNER)
            raise ValueError(msg)

    def _gather_infos(self):
        # Compute expert plan.
        if self.expert_type == AlfredExpertType.HANDCODED:
            self.state["extra.expert_plan"] = ["look"]
            try:
                # initialization
                if not self.prev_command:
                    self._handcoded_expert.observe(self.state["feedback"])
                else:
                    handcoded_expert_next_action = self._handcoded_expert.act(self.state, 0, self.state["won"], self.prev_command)
                    if handcoded_expert_next_action in self.state["admissible_commands"]:
                        self.state["extra.expert_plan"] = [handcoded_expert_next_action]
            except HandCodedAgentTimeout:
                raise Exception("Timeout")
        elif self.expert_type == AlfredExpertType.PLANNER:
            self.state["extra.expert_plan"] = self.state["policy_commands"]
        else:
            raise NotImplementedError("Unknown type of AlfredExpert: {}.".format(self.expert_type))

    def load(self, gamefile):
        super().load(gamefile)
        self.gamefile = gamefile
        self.request_infos.policy_commands = self.request_infos.policy_commands or (self.expert_type == AlfredExpertType.PLANNER)
        self.request_infos.facts = self.request_infos.facts or (self.expert_type == AlfredExpertType.HANDCODED)
        self._handcoded_expert = HandCodedTWAgent(max_steps=200)

    def step(self, command):
        self.state, reward, done = super().step(command)
        self.prev_command = str(command)
        self._gather_infos()
        return self.state, reward, done

    def reset(self):
        self.state = super().reset()
        self._handcoded_expert.reset(self.gamefile)
        self.prev_command = ""
        self._gather_infos()
        return self.state


class AlfredTWEnv(object):
    '''
    Interface for Textworld Env
    '''

    def __init__(self, config, train_eval="train", client_id=None, client_num=None, partition_strategy='uniform', min_games_per_client=100, start_idx=None, end_idx=None, **partition_kwargs):
        print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval
        self.client_id = client_id
        self.client_num = client_num
        self.partition_strategy = partition_strategy
        self.min_games_per_client = min_games_per_client
        self.partition_kwargs = partition_kwargs  # Extra parameters for the partition strategy
        self.start_idx = start_idx
        self.end_idx = end_idx
        if config["env"]["goal_desc_human_anns_prob"] > 0:
            msg = ("Warning! Changing `goal_desc_human_anns_prob` should be done with"
                   " the script `alfworld-generate`. Ignoring it and loading games as they are.")
            print(colored(msg, "yellow"))

        self.collect_game_files()
        self.use_expert = False
        print(f"use_expert = {self.use_expert}")
    def collect_game_files(self, verbose=False):
        def log(info):
            if verbose:
                print(info)

        self.game_files = []

        if self.train_eval == "train":
            data_path = os.path.expandvars(self.config['dataset']['data_path'])
        elif self.train_eval == "eval_in_distribution" and self.start_idx is not None and self.end_idx is not None:
            # start_idx/end_idx slice a contiguous window of game_files (see the eval
            # branch below). By default that window is over the TRAIN games (used by
            # eval/batch_alfworld_eval.sh SPLIT=train to collect hardness trajectories
            # over the whole training split). Set ALFWORLD_VAL_WINDOW=1 to window the
            # held-out valid_seen split instead, so the validation set can be covered
            # in contiguous batches (SPLIT=val batched eval). This flag is inert unless
            # explicitly set, so normal training/eval is unaffected.
            if os.getenv('ALFWORLD_VAL_WINDOW', '0') == '1':
                data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
                print(f"Using start_idx/end_idx to window the held-out valid_seen split")
            else:
                data_path = os.path.expandvars(self.config['dataset']['data_path'])
                print(f"Using start_idx and end_idx for eval_in_distribution (TRAIN games)")
        elif self.train_eval == "eval_in_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
        elif self.train_eval == "eval_out_of_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])

        log("Collecting solvable games...")

        # get task types
        assert len(self.config['env']['task_types']) > 0
        task_types = []
        for tt_id in self.config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        count = 0
        for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
            if 'traj_data.json' in files:
                count += 1

                # Filenames
                json_path = os.path.join(root, 'traj_data.json')
                game_file_path = os.path.join(root, "game.tw-pddl")

                if 'movable' in root or 'Sliced' in root:
                    log("Movable & slice trajs not supported %s" % (root))
                    continue

                # Get goal description
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    log("Skipping task type")
                    continue

                # Check if a game file exists
                if not os.path.exists(game_file_path):
                    log(f"Skipping missing game! {game_file_path}")
                    continue

                with open(game_file_path, 'r') as f:
                    gamedata = json.load(f)

                # Check if previously checked if solvable
                if 'solvable' not in gamedata:
                    print(f"-> Skipping missing solvable key! {game_file_path}")
                    continue

                if not gamedata['solvable']:
                    log("Skipping known %s, unsolvable game!" % game_file_path)
                    continue

                # Add to game file list
                self.game_files.append(game_file_path)

        # print(f"Overall we have {len(self.game_files)} games in split={self.train_eval}")
        original_num_games = len(self.game_files)

        shuffle_seed = self.partition_kwargs.get('shuffle_seed', 42)
        print(f"shuffle_seed: {shuffle_seed}")
        # shuffle the game_files
        if self.train_eval == "train":
            random.seed(shuffle_seed)
            random.shuffle(self.game_files)
        else:
            random.seed(42)
            random.shuffle(self.game_files)
        
        # ========== Federated learning data sharding ==========
        # breakpoint()
        if self.client_id is not None and self.client_num is not None:
            # Shard data only during training; evaluation uses the same full dataset
            if self.train_eval == "train":
                self.game_files = self.slice_games_for_client(self.game_files)
                if self.client_num == 1:
                    print(f"[Federated ALFWorld {self.train_eval.upper()}] Client {self.client_id}/{self.client_num}: "
                          f"Using {len(self.game_files)}/{original_num_games} games (TRAINING - NO SHARDING)")
                else:
                    print(f"[Federated ALFWorld {self.train_eval.upper()}] Client {self.client_id}/{self.client_num}: "
                          f"Using {len(self.game_files)}/{original_num_games} games (TRAINING - SHARDED)")
            else:
                # Evaluation uses the full dataset so results are comparable across all clients
                print(f"[Federated ALFWorld {self.train_eval.upper()}] Client {self.client_id}/{self.client_num}: "
                      f"Using {len(self.game_files)}/{original_num_games} games (EVALUATION - FULL DATASET)")
        else:
            print(f"[Standard ALFWorld {self.train_eval.upper()}] Using all {len(self.game_files)} games")
        self.num_games = len(self.game_files)

        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.game_files)
            print(f"num_train_games: {num_train_games}")
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
            print("Training with %d games" % (len(self.game_files)))
        else:
            # Evaluation uses a fixed dataset size so results are comparable across all clients
            # The evaluation dataset size can be controlled via config; defaults to 500
            # Priority: environment variable > config file > default value

                    # Handle the start_idx and end_idx parameters for batch inference
            if self.start_idx is not None and self.end_idx is not None:
                start_idx = self.start_idx
                end_idx = self.end_idx
                
                if start_idx is not None and end_idx is not None:
                    # Ensure the indices are within the valid range
                    start_idx = max(0, start_idx)
                    end_idx = min(end_idx, len(self.game_files))
                    
                    if start_idx < end_idx:
                        self.game_files = self.game_files[start_idx:end_idx]
                        
                        self.num_games = len(self.game_files)
                        print(f"[ALFWorld Batch Inference] Using games {start_idx}-{end_idx-1} (total: {self.num_games} games)")

                        # Detect duplicate game files
                        self._check_duplicate_games()
                    else:
                        print(f"[ALFWorld Batch Inference] Invalid range: start_idx={start_idx}, end_idx={end_idx}")
                        self.game_files = []
                        self.num_games = 0
            else:
                num_eval_games = os.getenv('ALFWORLD_EVAL_GAMES', None)
                if num_eval_games is not None:
                    num_eval_games = int(num_eval_games)
                else:
                    num_eval_games = self.config['dataset'].get('num_eval_games', 500)
                
                if num_eval_games > 0 and num_eval_games < len(self.game_files):

                    
                    self.game_files = random.sample(self.game_files, num_eval_games)
                    
                else:
                    # If configured as 0 or larger than the total data size, use all data
                    num_eval_games = len(self.game_files)

                self.num_games = len(self.game_files)
                print(f"Evaluating with {len(self.game_files)} games (fixed evaluation set for all clients)")

                # Verify the consistency of the evaluation dataset
                self._verify_evaluation_consistency()
        


    def _check_duplicate_games(self):
        """Detect whether there are duplicates among the game files"""
        if not self.game_files:
            print("[ALFWorld Duplicate Check] No games to check")
            return

        # Extract an identifier for each game file (keep the last two path segments)
        game_identifiers = []
        for game_file in self.game_files:
            # Extract the game identifier from the full path
            # e.g.: ~/.cache/alfworld/json_2.1.1/train/pick_two_obj_and_place-PepperShaker-None-Drawer-15/trial_T20190909_051727_474470/game.tw-pddl
            # should be extracted as: pick_two_obj_and_place-PepperShaker-None-Drawer-15/trial_T20190909_051727_474470
            if '/' in game_file:
                path_parts = game_file.split('/')
                if len(path_parts) >= 2:
                    # Take the last two segments (stripping the .tw-pddl suffix)
                    last_part = path_parts[-1]
                    if last_part.endswith('.tw-pddl'):
                        last_part = last_part[:-8]  # Strip '.tw-pddl'

                    second_last_part = path_parts[-2]
                    identifier = f"{second_last_part}/{last_part}"
                else:
                    # If there are not enough path segments, use the file name
                    identifier = path_parts[-1]
                    if identifier.endswith('.tw-pddl'):
                        identifier = identifier[:-8]
            else:
                # If there is no path separator, use the file name directly
                identifier = game_file
                if identifier.endswith('.tw-pddl'):
                    identifier = identifier[:-8]

            game_identifiers.append(identifier)

        # Detect duplicates
        from collections import Counter
        identifier_counts = Counter(game_identifiers)
        duplicates = {identifier: count for identifier, count in identifier_counts.items() if count > 1}
        
        if duplicates:
            print(f"[ALFWorld Duplicate Check] WARNING: Found {len(duplicates)} duplicate game identifiers:")
            for identifier, count in duplicates.items():
                print(f"  - '{identifier}': appears {count} times")
            
            # Show the specific paths of the duplicate games
            print("[ALFWorld Duplicate Check] Duplicate game files:")
            for identifier, count in duplicates.items():
                if count > 1:
                    # Find the matching game files
                    matching_files = []
                    for game_file in self.game_files:
                        # Extract the identifier of the current file
                        if '/' in game_file:
                            path_parts = game_file.split('/')
                            if len(path_parts) >= 2:
                                last_part = path_parts[-1]
                                if last_part.endswith('.tw-pddl'):
                                    last_part = last_part[:-8]
                                second_last_part = path_parts[-2]
                                file_identifier = f"{second_last_part}/{last_part}"
                            else:
                                file_identifier = path_parts[-1].replace('.tw-pddl', '')
                        else:
                            file_identifier = game_file.replace('.tw-pddl', '')
                        
                        if file_identifier == identifier:
                            matching_files.append(game_file)
                    
                    for i, file_path in enumerate(matching_files):
                        print(f"    {i+1}. {file_path}")
        else:
            print(f"[ALFWorld Duplicate Check] No duplicates found in {len(self.game_files)} games")
        
        # Summary statistics
        unique_games = len(set(game_identifiers))
        total_games = len(game_identifiers)
        if unique_games != total_games:
            print(f"[ALFWorld Duplicate Check] Summary: {unique_games} unique games out of {total_games} total games")
        else:
            print(f"[ALFWorld Duplicate Check] Summary: All {total_games} games are unique")

    
    def slice_games_for_client(self, game_files):
        """Shard the game files for a specific client, using the unified partition strategy module"""
        # Check data sufficiency
        total_games = len(game_files)
        total_min_required = self.min_games_per_client * self.client_num  # Minimum total number of games required across all clients
        
        if total_games < total_min_required:
            print(f"[ALFWorld] WARNING: Insufficient total games! Need at least {total_min_required} for {self.client_num} clients, have {total_games}")
            print(f"[ALFWorld] This may result in clients not getting enough games for training")
        else:
            print(f"[ALFWorld] Data sufficiency check passed: {total_games} games available for {self.client_num} clients")
            print(f"[ALFWorld] Each client can get at least {total_games // self.client_num} games")
        
        # Note: ALFWorld's client_id is actually 0-based, so no conversion is needed
        # But for consistency with other environments, we check whether conversion is required
        # if self.client_id is not None and self.client_id > 0:
        #     client_id_zero_based = self.client_id - 1
        # else:
        #     client_id_zero_based = self.client_id
        client_id_zero_based = self.client_id
        # Use the unified partition strategy module to split the data
        if self.partition_strategy == 'uniform':
            print(f"[DEBUG] ALFWorld partition: game_files={len(game_files)}, client_id={self.client_id}, client_id_zero_based={client_id_zero_based}, client_num={self.client_num}, min_games_per_client={self.min_games_per_client}")
            
            result = partition_dataset(
                data=game_files,
                strategy=self.partition_strategy,
                client_id=client_id_zero_based,
                client_num=self.client_num,
                min_samples_per_client=self.min_games_per_client,
                start_idx=0,  # ALFWorld starts at index 0 and has no validation set
                data_type='alfworld'
                # The uniform strategy does not require extra parameters
            )
            # The uniform strategy returns (data_slice, start_slice, end_slice)
            client_games_slice, start_slice, end_slice = result

            print(f"[DEBUG] ALFWorld partition result: start_slice={start_slice}, end_slice={end_slice}, client_games_slice_length={len(client_games_slice)}")

            # Use the data object directly
            self.game_files = client_games_slice
        elif self.partition_strategy == 'preference':
            # The preference strategy returns a data object; use it directly
            client_games_slice = partition_dataset(
                data=game_files,
                strategy=self.partition_strategy,
                client_id=client_id_zero_based,
                client_num=self.client_num,
                min_samples_per_client=self.min_games_per_client,
                start_idx=0,  # ALFWorld starts at index 0 and has no validation set
                data_type='alfworld',
                **self.partition_kwargs  # Pass through the partition strategy parameters
            )
            self.game_files = client_games_slice

            # Generate visualization images for the preference strategy
            try:
                import os

                # Get the output directory for the current client
                client_output_dir = os.environ.get('FEDERATED_OUTPUT_DIR', './output')
                round_num = os.environ.get('ROUND_NUM', 'unknown')
                client_id = os.environ.get('CLIENT_ID', 'unknown')

                # Create the client directory
                client_dir = os.path.join(client_output_dir, f'round_{round_num}', f'client_{client_id}')
                os.makedirs(client_dir, exist_ok=True)

                # Generate the visualization image
                tau_value = self.partition_kwargs.get('tau', 'unknown')
                viz_path = os.path.join(client_dir, f'alfworld_category_distribution_tau_{tau_value}.png')

                # Use the visualization function from partition_strategy
                visualize_alfworld_client_category_distribution(
                    client_games_slice=client_games_slice,
                    client_id=self.client_id,
                    client_num=self.client_num,
                    tau=tau_value,
                    save_path=viz_path
                )
                print(f"[ALFWorld Category Visualization] Saved to: {viz_path}")
                
                # Generate the distribution plot for all clients (generated by every client)
                all_clients_viz_path = os.path.join(client_dir, f'alfworld_all_clients_category_distribution_tau_{tau_value}.png')
                try:
                    # Get the original list of game files (before sharding)
                    original_game_files = self.game_files if hasattr(self, 'original_game_files') else game_files
                    
                    visualize_all_clients_category_distribution(
                        data=original_game_files,
                        client_num=self.client_num,
                        min_samples_per_client=self.min_games_per_client,
                        strategy='preference',
                        start_idx=0,
                        tau=tau_value,
                        save_path=all_clients_viz_path,
                        data_type='alfworld'
                    )
                    print(f"[ALFWorld All Clients Visualization] Saved to: {all_clients_viz_path}")
                except Exception as e:
                    print(f"[ALFWorld All Clients Visualization] Failed: {e}")
                
            except Exception as e:
                print(f"[ALFWorld Category Visualization] Failed to generate visualization: {e}")
        elif self.partition_strategy == 'coverage':
            print(f"[DEBUG] ALFWorld AlfredTWEnv: partition_strategy={self.partition_strategy}")
            print(f"[DEBUG] ALFWorld AlfredTWEnv: partition_kwargs={self.partition_kwargs}")
            client_games_slice = partition_dataset(
                data=game_files,
                strategy=self.partition_strategy,
                client_id=client_id_zero_based,
                client_num=self.client_num,
                min_samples_per_client=self.min_games_per_client,
                start_idx=0,
                data_type='alfworld',
                **self.partition_kwargs  # Pass through the partition strategy parameters
            )
            self.game_files = client_games_slice
             # Generate the visualization charts for the coverage strategy
            try:
                import os

                # Get the output directory for the current client
                client_output_dir = os.environ.get('FEDERATED_OUTPUT_DIR', './output')
                round_num = os.environ.get('ROUND_NUM', 'unknown')
                client_id = os.environ.get('CLIENT_ID', 'unknown')

                # Create the client directory
                client_dir = os.path.join(client_output_dir, f'round_{round_num}', f'client_{client_id}')
                os.makedirs(client_dir, exist_ok=True)

                # Generate the visualization image for the coverage strategy
                if 'size_std' not in self.partition_kwargs:
                    raise ValueError("Missing required 'size_std' parameter in partition_kwargs for coverage visualization.")
                size_std_value = self.partition_kwargs['size_std']
                viz_path = os.path.join(client_dir, f'coverage_distribution_std_{size_std_value}.png')
                
                visualize_coverage_normal_distribution(
                    data=game_files,
                    client_num=self.client_num,
                    min_samples_per_client=self.min_games_per_client,
                    start_idx=0,
                    **self.partition_kwargs,
                    save_path=viz_path
                )
                print(f"[ALFWorld Coverage Visualization] Saved to: {viz_path}")
                

                    
            except Exception as e:
                print(f"[ALFWorld Coverage Visualization] Failed to generate visualization: {e}")
        elif self.partition_strategy == 'hardness':
            client_games_slice = partition_dataset(
                data=game_files,
                strategy=self.partition_strategy,
                client_id=client_id_zero_based,
                client_num=self.client_num,
                min_samples_per_client=self.min_games_per_client,
                start_idx=0,
                data_type='alfworld',
                **self.partition_kwargs
            )
            self.game_files = client_games_slice

            # Generate the visualization for the hardness strategy (only on the first client)

            try:
                import os
                from agent_system.environments.partition_strategy import visualize_hardness_distribution_alfworld

                # Get the output directory for the current client
                client_output_dir = os.environ.get('FEDERATED_OUTPUT_DIR', './output')
                round_num = os.environ.get('ROUND_NUM', 'unknown')
                client_id = os.environ.get('CLIENT_ID', 'unknown')

                # Create the client directory
                client_dir = os.path.join(client_output_dir, f'round_{round_num}', f'client_{client_id}')
                os.makedirs(client_dir, exist_ok=True)

                # Generate the visualization image
                success_std = self.partition_kwargs.get('success_std', 0.1)
                save_path = os.path.join(client_dir, f'hardness_distribution_alfworld_std_{success_std}.png')
                
                visualize_hardness_distribution_alfworld(
                    data=game_files,
                    client_num=self.client_num,
                    min_samples_per_client=self.min_games_per_client,
                    start_idx=0,
                    save_path=save_path,
                    **self.partition_kwargs
                )
                
            except Exception as e:
                print(f"[ALFWorld Hardness Visualization] Failed to generate visualization: {e}")
        elif self.partition_strategy == 'env_disjoint':
            # Env-level heterogeneity: scene-disjoint partition
            # See docs/heterogeneity.md
            client_games_slice = partition_dataset(
                data=game_files,
                strategy=self.partition_strategy,
                client_id=client_id_zero_based,
                client_num=self.client_num,
                min_samples_per_client=self.min_games_per_client,
                start_idx=0,
                data_type='alfworld',
                **self.partition_kwargs
            )
            self.game_files = client_games_slice

            # Optional viz (per round, per client) — same pattern as category/coverage
            try:
                import os
                from agent_system.environments.partition_strategy import (
                    visualize_alfworld_env_partition,
                )
                client_output_dir = os.environ.get('FEDERATED_OUTPUT_DIR', './output')
                round_num = os.environ.get('ROUND_NUM', 'unknown')
                env_client_id = os.environ.get('CLIENT_ID', 'unknown')
                client_dir = os.path.join(client_output_dir, f'round_{round_num}',
                                          f'client_{env_client_id}')
                os.makedirs(client_dir, exist_ok=True)
                env_div_value = self.partition_kwargs.get('env_div', 0.7)
                fallback_value = self.partition_kwargs.get('fallback', 'skip')
                viz_path = os.path.join(
                    client_dir,
                    f'env_disjoint_distribution_div_{env_div_value}_fb_{fallback_value}.png'
                )
                # Per-client viz uses the SAME game_files pool & partition kwargs;
                # only the sample being highlighted is this client's.
                # We skip the full N-client matrix here for efficiency
                # (visualize_alfworld_env_partition runs partition() N times).
                # Just log the slice state.
                print(f"[ENV-AlfWorld viz] (skipped per-round full matrix viz; "
                      f"run tools/env_heterogeneity/viz_alfworld_partition.py offline)")
            except Exception as e:
                print(f"[ENV-AlfWorld] viz hook failed: {e}")
        else:
            raise ValueError(f"Invalid partition strategy: {self.partition_strategy}. "
                             f"Supported: uniform, preference, coverage, hardness, env_disjoint")
        
        # Get the partition info for log output
        partition_info = get_partition_info(
            data=game_files,
            strategy=self.partition_strategy,
            client_id=client_id_zero_based,
            client_num=self.client_num,
            min_samples_per_client=self.min_games_per_client,
            start_idx=0,
            data_type='alfworld',
            **self.partition_kwargs
        )
        
        if self.partition_strategy == 'uniform':
            print(f"[ALFWorld] Client {self.client_id}/{self.client_num} ({self.partition_strategy}): "
                  f"Games {partition_info['start_idx']}-{partition_info['end_idx']-1} "
                  f"(total: {partition_info['actual_samples']} games, min: {self.min_games_per_client})")
        elif self.partition_strategy == 'preference':
            tau_value = self.partition_kwargs.get('tau', 'unknown')
            print(f"[ALFWorld] Client {self.client_id}/{self.client_num} ({self.partition_strategy}, tau={tau_value}): "
                  f"Total: {partition_info['actual_samples']} games, min: {self.min_games_per_client}")
        elif self.partition_strategy == 'coverage':
            coverage_value = self.partition_kwargs.get('coverage', 'unknown')
            print(f"[ALFWorld] Client {self.client_id}/{self.client_num} ({self.partition_strategy}, coverage={coverage_value}): "
                  f"Total: {partition_info['actual_samples']} games, min: {self.min_games_per_client}")
        else:
            print(f"[ALFWorld] Client {self.client_id}/{self.client_num} ({self.partition_strategy}): "
                  f"Total: {partition_info['actual_samples']} games, min: {self.min_games_per_client}")
        
        return self.game_files


    def _get_task_priority_order(self, games_by_task):
        """Dynamically determine task type priority based on the number of available games and training importance"""
        # Base priority (can be adjusted to fit actual needs)
        base_priority = [
            'pick_and_place_simple',           # Priority 1: simple task, basic skill
            'pick_two_obj_and_place',          # Priority 2: two-object task, medium complexity
            'pick_clean_then_place_in_recep',  # Priority 3: cleaning task, practical skill
            'pick_heat_then_place_in_recep',   # Priority 4: heating task, practical skill
            'pick_cool_then_place_in_recep',   # Priority 5: cooling task, practical skill
            'look_at_obj_in_light'             # Priority 6: observation task, basic skill
        ]

        # Adjust priority based on the number of available games
        # Prefer task types that have more available games
        available_games_count = {}
        for task_type, games in games_by_task.items():
            # Compute how many games this task type can still provide to the current client
            total_games = len(games)
            # Estimate the number of games already taken by other clients
            estimated_used = (self.client_num - 1) * (total_games // self.client_num)
            available_games_count[task_type] = total_games - estimated_used

        # Sort by the number of available games; more games means higher priority
        sorted_by_availability = sorted(
            base_priority, 
            key=lambda x: available_games_count.get(x, 0), 
            reverse=True
        )
        
        print(f"[ALFWorld] Task priority order (by availability): {sorted_by_availability}")
        for task_type in sorted_by_availability:
            if task_type in available_games_count:
                print(f"  {task_type}: {available_games_count[task_type]} available games")
        
        return sorted_by_availability

    def _print_final_distribution_report(self, client_games, games_by_task):
        """Print the final distribution statistics report"""
        print(f"\n{'='*60}")
        print(f"[ALFWorld] FINAL DISTRIBUTION REPORT for Client {self.client_id}")
        print(f"{'='*60}")

        # Tally by task type
        task_distribution = {}
        for game in client_games:
            task_type = self.extract_task_type_from_path(game)
            if task_type not in task_distribution:
                task_distribution[task_type] = 0
            task_distribution[task_type] += 1

        # Print the task type distribution
        print(f"Task Type Distribution:")
        total_games = len(client_games)
        for task_type in sorted(task_distribution.keys()):
            count = task_distribution[task_type]
            percentage = (count / total_games) * 100 if total_games > 0 else 0
            print(f"  {task_type}: {count} games ({percentage:.1f}%)")

        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total games: {total_games}")
        print(f"  Target games: 100")
        print(f"  Achievement: {(total_games/100)*100:.1f}%")

        # Assessment and recommendation
        if total_games >= 100:
            print(f"  Status: ✅ SUCCESS - Target achieved!")
        elif total_games >= 80:
            print(f"  Status: ⚠️  WARNING - Close to target ({total_games}/100)")
        elif total_games >= 50:
            print(f"  Status: ⚠️  WARNING - Below target but usable ({total_games}/100)")
        else:
            print(f"  Status: ❌ CRITICAL - Insufficient games ({total_games}/100)")
            print(f"  Recommendation: Consider reducing client_num or increasing dataset size")
        
        print(f"{'='*60}\n")

    def _verify_evaluation_consistency(self):
        """Verify the consistency of the evaluation dataset, ensuring all clients use the same evaluation data"""
        if self.train_eval != "train" and self.client_id is not None:
            # Compute the hash of the evaluation dataset to verify consistency
            import hashlib
            sorted_games = sorted(self.game_files)
            games_hash = hashlib.md5(''.join(sorted_games).encode()).hexdigest()[:8]

            print(f"[ALFWorld] EVALUATION CONSISTENCY CHECK:")
            print(f"  Client {self.client_id}: Evaluation dataset hash: {games_hash}")
            print(f"  Total evaluation games: {len(self.game_files)}")

            # Check the task type distribution
            task_distribution = {}
            for game in self.game_files:
                task_type = self.extract_task_type_from_path(game)
                if task_type not in task_distribution:
                    task_distribution[task_type] = 0
                task_distribution[task_type] += 1
            
            print(f"  Evaluation task distribution:")
            for task_type in sorted(task_distribution.keys()):
                count = task_distribution[task_type]
                percentage = (count / len(self.game_files)) * 100
                print(f"    {task_type}: {count} games ({percentage:.1f}%)")
            
            print(f"  ✅ All clients will use the same evaluation dataset (hash: {games_hash})")

    def extract_task_type_from_path(self, game_file_path):
        """Extract the task type from the game file path"""
        # ALFWorld paths usually contain task type information
        # e.g.: .../pick_and_place_simple-Mug-None-CoffeeTable-7/trial_T20190907_140408_276392/
        path_parts = game_file_path.split(os.sep)

        for part in path_parts:
            for task_id, task_name in TASK_TYPES.items():
                if task_name in part:
                    return task_name

        # Default to returning the first task type
        return list(TASK_TYPES.values())[0]
    def get_game_logic(self):
        self.game_logic = {
            "pddl_domain": open(os.path.expandvars(self.config['logic']['domain'])).read(),
            "grammar": open(os.path.expandvars(self.config['logic']['grammar'])).read()
        }

    # use expert to check the game is solvable
    def is_solvable(self, env, game_file_path,
                    random_perturb=True, random_start=10, random_prob_after_state=0.15):
        done = False
        steps = 0
        trajectory = []
        try:
            env.load(game_file_path)
            game_state = env.reset()
            if env.expert_type == AlfredExpertType.PLANNER:
                return game_state["extra.expert_plan"]

            while not done:
                expert_action = game_state['extra.expert_plan'][0]
                random_action = random.choice(game_state.admissible_commands)

                command = expert_action
                if random_perturb:
                    if steps <= random_start or random.random() < random_prob_after_state:
                        command = random_action

                game_state, _, done = env.step(command)
                trajectory.append(command)
                steps += 1
        except Exception as e:
            print("Unsolvable: %s (%s)" % (str(e), game_file_path))
            return None

        return trajectory

    def init_env(self, batch_size, worker_idx=None):
        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        # Register a new Gym environment.
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = self.config["env"]["expert_type"]
        training_method = self.config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]
            if self.use_expert:
                expert_plan = True if self.train_eval == "train" else False
            else:
                expert_plan = False
            if expert_plan:
                wrappers.append(AlfredExpert(expert_type))
                request_infos.extras.append("expert_plan")

        else:
            raise NotImplementedError

        # For validation workers, select specific game by worker index
        game_files_to_use = self.game_files
        if worker_idx is not None and self.train_eval != "train":
            # For validation, each worker should get a specific game by index
            if worker_idx < len(self.game_files):
                game_files_to_use = [self.game_files[worker_idx]]
                print(f"[ALFWorld Worker {worker_idx}] Using specific game: {self.game_files[worker_idx]}")
            else:
                print(f"[ALFWorld Worker {worker_idx}] Warning: worker_idx {worker_idx} >= num_games {len(self.game_files)}, using random selection")
                # Fall back to random selection if worker_idx is out of range
                game_files_to_use = [random.choice(self.game_files)]

        env_id = textworld.gym.register_games(game_files_to_use, request_infos,
                                              batch_size=batch_size,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode,
                                              wrappers=wrappers)
        
        # env_id = textworld.gym.register_games(self.game_files, request_infos,
        #                                       batch_size=batch_size,
        #                                       asynchronous=True,
        #                                       max_episode_steps=max_nb_steps_per_episode,
        #                                       wrappers=wrappers)
        # Launch Gym environment.
        env = textworld.gym.make(env_id)
        return env
