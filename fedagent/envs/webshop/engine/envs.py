import ray
import gym
import numpy as np
from ...partition_strategy import partition_dataset, get_partition_info, visualize_webshop_client_category_distribution, visualize_all_clients_category_distribution, visualize_coverage_normal_distribution, visualize_hardness_distribution

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

@ray.remote(num_cpus=0.2)
class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: dict = None,
        client_id: int = None,
        client_num: int = None,
        min_goals_per_client: int = 100,
        val_batch_size: int = 500,
        partition_strategy: str = 'uniform',
        infer_special: bool = False,
        start_idx: int = None,
        end_idx: int = None,
        **partition_kwargs
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1
        self.client_id = client_id
        self.client_num = client_num
        self.min_goals_per_client = min_goals_per_client
        self.val_batch_size = val_batch_size
        self.partition_strategy = partition_strategy
        self.infer_special = infer_special
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.partition_kwargs = partition_kwargs  # Extra keyword arguments for the partition strategy
        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        self._workers = []

        for i in range(self.num_processes):
            worker = WebshopWorker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)
        
        # Compute a task_id hash for each goal and dump them to a JSON file (used to detect duplicates)
        # import hashlib
        # import json
        # import os
        
        # goal_hashes = []
        # for i, goal in enumerate(goals):
        #     task_id = None
        #     if isinstance(goal, dict):
        #         if 'asin' in goal:
        #             asin = goal['asin']
        #             if 'goal_options' in goal and goal['goal_options']:
        #                 # For synthetic goals: use asin + goal_options hash
        #                 options_str = str(sorted(goal['goal_options'].items()))
        #                 options_hash = int(hashlib.md5(options_str.encode()).hexdigest(), 16)
        #                 task_id = f"{asin}_{abs(options_hash)}"
        #             else:
        #                 # Fallback to asin + instruction_text hash for human goals
        #                 if 'instruction_text' in goal:
        #                     instruction_hash = int(hashlib.md5(goal['instruction_text'].encode()).hexdigest(), 16)
        #                     task_id = f"{asin}_{abs(instruction_hash)}"
        #                 else:
        #                     task_id = asin
            
        #     goal_hashes.append({
        #         'index': i,
        #         'task_id': task_id,
        #         'goal': goal

        #     })
        
        # # Save to a JSON file
        # hash_file = "goal_hashes.json"
        # with open(hash_file, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         'total_goals': len(goals),
        #         'goal_hashes': goal_hashes
        #     }, f, indent=2, ensure_ascii=False)
        
        # print(f"[DEBUG] Saved {len(goals)} goal hashes to {hash_file}")
        
        # # Check for duplicate task_ids
        # task_ids = [gh['task_id'] for gh in goal_hashes if gh['task_id'] is not None]
        # unique_task_ids = set(task_ids)
        # duplicate_count = len(task_ids) - len(unique_task_ids)

        # if duplicate_count > 0:
        #     print(f"[WARNING] Found {duplicate_count} duplicate task_ids in goals!")
        #     # Identify which task_ids are duplicated
        #     from collections import Counter
        #     task_id_counts = Counter(task_ids)
        #     duplicates = {tid: count for tid, count in task_id_counts.items() if count > 1}
        #     print(f"[WARNING] Duplicate task_ids: {duplicates}")
        # else:
        #     print(f"[INFO] No duplicate task_ids found in {len(goals)} goals")

        


        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        # if not self.is_train:
        #     self.goal_idxs = range(500)
        # else:
        #     self.goal_idxs = range(500, len(goals))
            
        # print(self.goal_idxs)

        # -------goal slicing for federated learning----------
        # breakpoint()
        if not self.is_train:
            # The validation set is never sharded; every client evaluates on the same validation set
            if self.infer_special:
                # Special inference mode: use the explicitly provided index range
                if self.start_idx is not None and self.end_idx is not None:
                    # Use the explicitly provided index range (offset by 500 to skip the val pool)
                    start_idx = self.start_idx+500
                    end_idx = self.end_idx+500
                else:
                    # Default: use the goals located after index 500
                    start_idx = 500
                    end_idx = start_idx + self.val_batch_size
                
                total_val_goals = list(range(start_idx, end_idx))
                self.goal_idxs = total_val_goals
                
                if self.client_id is not None and self.client_num is not None:
                    print(f"[WebShop infer_special] Client {client_id}/{client_num}: "
                        f"rolling out TRAINING-pool goals {start_idx}-{end_idx-1} (idx 500+ = train; for trajectory collection, NOT held-out validation) - NO SHARDING")
                else:
                    print(f"[WebShop infer_special] rolling out TRAINING-pool goals {start_idx}-{end_idx-1} (idx 500+ = train; for trajectory collection, NOT held-out validation)")
            else:
                # Standard validation set: the held-out pool goals[0:500].
                if self.start_idx is not None and self.end_idx is not None:
                    # Windowed held-out val (NO +500 offset, unlike infer_special):
                    # evaluate goals[start:end] WITHIN the val pool [0:500], so the
                    # full validation set can be covered in contiguous batches,
                    # symmetric with the training-pool batching above. start/end must
                    # lie in [0, 500]. This branch is inert during normal
                    # training/eval (start_idx/end_idx default to None there); it is
                    # only exercised by eval/batch_webshop_eval.sh SPLIT=val.
                    total_val_goals = list(range(self.start_idx, self.end_idx))
                else:
                    # Full held-out pool (val_batch_size defaults to 500).
                    total_val_goals = list(range(self.val_batch_size))
                self.goal_idxs = total_val_goals
                _idx_lo = total_val_goals[0] if total_val_goals else 'NA'
                _idx_hi = total_val_goals[-1] if total_val_goals else 'NA'
                if self.client_id is not None and self.client_num is not None:
                    print(f"[Federated WebShop VAL] Client {client_id}/{client_num}: "
                        f"Using {len(total_val_goals)} validation goals (idx {_idx_lo}..{_idx_hi}) - NO SHARDING")
                else:
                    print(f"[Standard WebShop VAL] Using {len(total_val_goals)} validation goals (idx {_idx_lo}..{_idx_hi})")
        else:
            # The training set is sharded per client, guaranteeing each client gets at least the specified number of goals
            if self.client_id is not None and self.client_num is not None:
                # Federated learning mode: use the shared partition-strategy module
                min_goals_per_client = self.min_goals_per_client

                # Split the data via the partition-strategy module
                if self.partition_strategy == 'uniform':
                    result = partition_dataset(
                        data=goals,
                        strategy=self.partition_strategy,
                        client_id=self.client_id,
                        client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500,  # Skip the validation set; start from index 500
                        data_type='webshop'  # Specify the data type as WebShop
                        # The uniform strategy needs no additional parameters
                    )
                    # The uniform strategy returns a tuple (data_slice, start_slice, end_slice)
                    client_goal_slice, start_slice, end_slice = result
                    # Use the index range directly
                    self.goal_idxs = [500 + i for i in range(start_slice, end_slice)]
                elif self.partition_strategy == 'preference':
                    # The preference strategy returns data objects, which must be converted back to indices
                    client_goal_slice = partition_dataset(
                        data=goals,
                        strategy=self.partition_strategy,
                        client_id=self.client_id,
                        client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500,  # Skip the validation set; start from index 500
                        data_type='webshop',  # Specify the data type as WebShop
                        **self.partition_kwargs  # Forward the partition parameters (including tau, etc.)
                    )

                    # Convert the data objects back to indices (also required for the preference strategy)
                    self.goal_idxs = []
                    print(f"[DEBUG] preference strategy: client_goal_slice length = {len(client_goal_slice)}")
                    for goal in client_goal_slice:
                        # Find the corresponding index in the original goals list
                        try:
                            idx = goals.index(goal)
                            self.goal_idxs.append(idx)
                        except ValueError:
                            # If it cannot be found, skip this goal
                            print(f"[DEBUG] Warning: could not find this goal's index in the original goals list")
                            continue
                    print(f"[DEBUG] preference strategy: self.goal_idxs length = {len(self.goal_idxs)}")

                    # Generate a visualization image for the preference strategy
                    if self.partition_strategy == 'preference':
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
                            if 'tau' not in self.partition_kwargs:
                                raise ValueError("Missing required 'tau' parameter in partition_kwargs for preference partition visualization.")
                            tau_value = self.partition_kwargs['tau']
                            viz_path = os.path.join(client_dir, f'category_distribution_tau_{tau_value}.png')
                            visualize_webshop_client_category_distribution(
                                client_goals_slice=client_goal_slice,
                                client_id=self.client_id,
                                client_num=self.client_num,
                                save_path=viz_path,
                                **self.partition_kwargs
                            )
                            print(f"[Category Visualization] Saved to: {viz_path}")

                            # Generate the distribution plot across all clients (emitted by every client)
                            if 'tau' not in self.partition_kwargs:
                                raise ValueError("Missing required 'tau' parameter in partition_kwargs for preference partition visualization.")
                            tau_value = self.partition_kwargs['tau']
                            all_clients_viz_path = os.path.join(client_dir, f'webshop_all_clients_category_distribution_tau_{tau_value}.png')
                            try:
                                # Get the original goals list (before sharding)
                                original_goals = goals

                                visualize_all_clients_category_distribution(
                                    data=original_goals,
                                    client_num=self.client_num,
                                    min_samples_per_client=min_goals_per_client,
                                    strategy='preference',
                                    start_idx=500,
                                    save_path=all_clients_viz_path,
                                    data_type='webshop',
                                    **self.partition_kwargs
                                )
                                print(f"[WebShop All Clients Visualization] Saved to: {all_clients_viz_path}")
                            except Exception as e:
                                print(f"[WebShop All Clients Visualization] Failed: {e}")
                        except Exception as e:
                            print(f"[Category Visualization] Failed to generate visualization: {e}")
                elif self.partition_strategy == 'coverage':
                    # The coverage strategy returns data objects, which must be converted back to indices
                    client_goal_slice = partition_dataset(
                        data=goals,
                        strategy=self.partition_strategy,
                        client_id=self.client_id,
                        client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500,  # Skip the validation set; start from index 500
                        data_type='webshop',  # Specify the data type as WebShop
                        **self.partition_kwargs  # Forward the coverage-strategy parameters (e.g. size_std)
                    )
                    # Convert the data objects back to indices (required for both the category and coverage strategies)
                    self.goal_idxs = []
                    print(f"[DEBUG] coverage strategy: client_goal_slice length = {len(client_goal_slice)}")
                    for goal in client_goal_slice:
                        # Find the corresponding index in the original goals list
                        try:
                            idx = goals.index(goal)
                            self.goal_idxs.append(idx)
                        except ValueError:
                            # If it cannot be found, skip this goal
                            print(f"[DEBUG] Warning: could not find this goal's index in the original goals list")
                            continue
                    print(f"[DEBUG] coverage strategy: self.goal_idxs length = {len(self.goal_idxs)}")
                    
                    # Generate the visualization plot for the coverage strategy
                    # try:
                    #     import os

                    #     # Get the output directory for the current client
                    #     client_output_dir = os.environ.get('FEDERATED_OUTPUT_DIR', './output')
                    #     round_num = os.environ.get('ROUND_NUM', 'unknown')
                    #     client_id = os.environ.get('CLIENT_ID', 'unknown')

                    #     # Create the client directory
                    #     client_dir = os.path.join(client_output_dir, f'round_{round_num}', f'client_{client_id}')
                    #     os.makedirs(client_dir, exist_ok=True)

                    #     # Generate the visualization image for the coverage strategy
                    #     if 'size_std' not in self.partition_kwargs:
                    #         raise ValueError("Missing required 'size_std' parameter in partition_kwargs for coverage visualization.")
                    #     size_std_value = self.partition_kwargs['size_std']
                    #     viz_path = os.path.join(client_dir, f'coverage_distribution_std_{size_std_value}.png')
                        
                    #     visualize_coverage_normal_distribution(
                    #         data=goals,
                    #         client_num=self.client_num,
                    #         min_samples_per_client=min_goals_per_client,
                    #         start_idx=500,
                    #         **self.partition_kwargs,
                    #         save_path=viz_path
                    #     )
                    #     print(f"[Coverage Visualization] Saved to: {viz_path}")
                        

                            
                    # except Exception as e:
                    #     print(f"[Coverage Visualization] Failed to generate visualization: {e}")
                elif self.partition_strategy == 'hardness':
                    if 'success_std' not in self.partition_kwargs:
                        raise ValueError("Missing required 'success_std' parameter in partition_kwargs for hardness partition strategy.")
                    success_std = self.partition_kwargs['success_std']

                    # The hardness strategy returns data objects, which must be converted back to indices
                    client_goal_slice = partition_dataset(
                        data=goals,
                        strategy=self.partition_strategy,
                        client_id=self.client_id,
                        client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500,  # Skip the validation set; start from index 500
                        data_type='webshop',  # Specify the data type as WebShop
                        **self.partition_kwargs  # Forward the hardness-strategy parameters
                    )
                    # Convert the data objects back to indices
                    self.goal_idxs = []
                    print(f"[DEBUG] hardness strategy: client_goal_slice length = {len(client_goal_slice)}")
                    for goal in client_goal_slice:
                        # Find the corresponding index in the original goals list
                        try:
                            idx = goals.index(goal)
                            self.goal_idxs.append(idx)
                        except ValueError:
                            # If it cannot be found, skip this goal
                            print(f"[DEBUG] Warning: could not find this goal's index in the original goals list")
                            continue
                    print(f"[DEBUG] hardness strategy: self.goal_idxs length = {len(self.goal_idxs)}")
                    
                    # Generate the visualization plot for the hardness strategy
                    # try:
                    #     import os

                    #     # Get the output directory for the current client
                    #     client_output_dir = os.environ.get('FEDERATED_OUTPUT_DIR', './output')
                    #     round_num = os.environ.get('ROUND_NUM', 'unknown')
                    #     client_id = os.environ.get('CLIENT_ID', 'unknown')

                    #     # Create the client directory
                    #     client_dir = os.path.join(client_output_dir, f'round_{round_num}', f'client_{client_id}')
                    #     os.makedirs(client_dir, exist_ok=True)

                    #     # Generate the visualization image for the hardness strategy
                    #     success_std_value = self.partition_kwargs.get('success_std', 0.1)
                    #     viz_path = os.path.join(client_dir, f'hardness_distribution_std_{success_std_value}.png')

                    #     # Call the hardness-strategy visualization function
                    #     visualize_hardness_distribution(
                    #         data=goals,
                    #         client_num=self.client_num,
                    #         min_samples_per_client=min_goals_per_client,
                    #         start_idx=500,
                    #         success_std=success_std_value,
                    #         save_path=viz_path
                    #     )
                    #     print(f"[Hardness Visualization] Saved to: {viz_path}")
                        
                    # except Exception as e:
                    #     print(f"[Hardness Visualization] Failed to generate visualization: {e}")
                elif self.partition_strategy == 'distractor_disjoint':
                    # Env-level heterogeneity: the catalog split is precomputed in fed_env_manager
                    # and written into env_kwargs['catalog_filter_asins'] (forwarded to SimServer's internal filter).
                    # Here goal_idxs is shared by all clients and spans the entire train pool (= goals[500:]).
                    # The heterogeneity lives in SimServer.product_item_dict, not in goal_idxs.
                    self.goal_idxs = list(range(500, len(goals)))
                    print(f"[Federated WebShop TRAIN env-level] Client {self.client_id}/{self.client_num} "
                          f"(distractor_disjoint): goal_idxs=[500:{len(goals)}], "
                          f"total {len(self.goal_idxs)} train goals (all clients share). "
                          f"Heterogeneity is on catalog axis (see SimServer.catalog_filter_asins).")
                elif self.partition_strategy == 'catalog_split':
                    # PAPER VARIANT 1: Catalog Split (Environment-Level heterogeneity,
                    # Stage 1 = content/catalog). code key 'catalog_split'.
                    # See docs/heterogeneity.md
                    # Both catalog (env axis) and goal_idxs (task axis) are computed in
                    # fed_env_manager.py via _distractor_disjoint_partition_webshop_v5.
                    # (The '_v5' suffix is the partition function's IMPLEMENTATION-REVISION
                    #  number, NOT the paper's Variant 5 / Rank Wrapper; v4 and v5 of this
                    #  function both realize paper Variant 1, Catalog Split.)
                    #   - env_kwargs['catalog_filter_asins'] -> SimServer catalog filter
                    #   - env_kwargs['client_goal_idxs']     -> self.goal_idxs (uniform 100/client)
                    if 'client_goal_idxs' in self._env_kwargs and self._env_kwargs['client_goal_idxs']:
                        self.goal_idxs = list(self._env_kwargs['client_goal_idxs'])
                        # validate range vs goals length
                        if max(self.goal_idxs) >= len(goals):
                            raise ValueError(
                                f"[Catalog-Split] client_goal_idxs max={max(self.goal_idxs)} "
                                f">= len(goals)={len(goals)}. partition mismatch?"
                            )
                    else:
                        # Fallback: client_goal_idxs not in env_kwargs (unexpected for catalog-split).
                        # Re-run uniform_partition here to recover; matches what
                        # fed_env_manager.py would have computed (deterministic).
                        result = partition_dataset(
                            data=goals, strategy='uniform',
                            client_id=self.client_id, client_num=self.client_num,
                            min_samples_per_client=min_goals_per_client,
                            start_idx=500, data_type='webshop',
                        )
                        _, start_slice, end_slice = result
                        self.goal_idxs = [500 + i for i in range(start_slice, end_slice)]
                    print(f"[Federated WebShop TRAIN env-level Catalog-Split] Client {self.client_id}/{self.client_num} "
                          f"(catalog_split): |goal_idxs|={len(self.goal_idxs)} "
                          f"(range {self.goal_idxs[0] if self.goal_idxs else 'N/A'}..{self.goal_idxs[-1] if self.goal_idxs else 'N/A'}), "
                          f"per-client target+catalog filter applied via SimServer.catalog_filter_asins.")
                elif self.partition_strategy in ('bm25_variant', 'lookalike_injection', 'rank_wrapper'):
                    # Transition-level env-heterogeneity (BM25 Reweighting / Lookalike Injection):
                    # task partition is uniform; env axis is heterogeneous via
                    # SimServer.bm25_in_memory_config (BM25 Reweighting) or .extra_products (Lookalike Injection).
                    # See docs/heterogeneity.md
                    result = partition_dataset(
                        data=goals, strategy='uniform',
                        client_id=self.client_id, client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500, data_type='webshop',
                    )
                    _, start_slice, end_slice = result
                    self.goal_idxs = [500 + i for i in range(start_slice, end_slice)]
                    print(f"[Federated WebShop TRAIN env-level transition] Client {self.client_id}/{self.client_num} "
                          f"({self.partition_strategy}): |goal_idxs|={len(self.goal_idxs)} "
                          f"(uniform task partition); env heterogeneity injected via SimServer kwargs.")
                else:
                    raise ValueError(f"Invalid partition strategy: {self.partition_strategy}. Supported strategies: uniform, preference, coverage, hardness, distractor_disjoint, catalog_split, bm25_variant, lookalike_injection, rank_wrapper")
                
                # Gather partition info for logging output
                if self.partition_strategy == 'distractor_disjoint':
                    # env-level: goals are not split; partition_info follows the uniform shape (the whole train pool)
                    partition_info = {
                        'start_idx': 0,
                        'end_idx': len(goals) - 500,
                        'actual_samples': len(self.goal_idxs),
                    }
                elif self.partition_strategy == 'catalog_split':
                    # catalog-split: task partition is uniform 100/client; goal_idxs is a client-specific contiguous slice
                    if self.goal_idxs:
                        partition_info = {
                            'start_idx': self.goal_idxs[0] - 500,
                            'end_idx': self.goal_idxs[-1] - 500 + 1,
                            'actual_samples': len(self.goal_idxs),
                        }
                    else:
                        partition_info = {'start_idx': 0, 'end_idx': 0, 'actual_samples': 0}
                elif self.partition_strategy in ('uniform', 'bm25_variant', 'lookalike_injection', 'rank_wrapper'):
                    # uniform: standard task partition
                    # bm25_variant / lookalike_injection: uniform task + env-only heterogeneity
                    partition_info = get_partition_info(
                        data=goals,
                        strategy='uniform',
                        client_id=self.client_id,
                        client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500,
                        data_type='webshop'
                    )
                else:
                    # The category, coverage, and hardness strategies require additional parameters
                    partition_info = get_partition_info(
                        data=goals,
                        strategy=self.partition_strategy,
                        client_id=self.client_id,
                        client_num=self.client_num,
                        min_samples_per_client=min_goals_per_client,
                        start_idx=500,
                        data_type='webshop',
                        **self.partition_kwargs
                    )
                
                if self.partition_strategy in ('uniform', 'bm25_variant', 'lookalike_injection', 'rank_wrapper'):
                    # uniform / bm25_variant / lookalike_injection share the same task partition layout;
                    # start_idx and end_idx are relative to start_idx=500.
                    actual_start = 500 + partition_info['start_idx']
                    actual_end = 500 + partition_info['end_idx'] - 1
                    print(f"[Federated WebShop TRAIN] Client {self.client_id}/{self.client_num} ({self.partition_strategy}): "
                        f"Train Goals {actual_start}-{actual_end} "
                        f"(total: {partition_info['actual_samples']} goals, min: {min_goals_per_client})")
                elif self.partition_strategy == 'preference':
                    # Analyze the category distribution for the current client
                    category_counts = {}
                    for goal_idx in self.goal_idxs:
                        if goal_idx < len(goals):
                            goal = goals[goal_idx]
                            category = goal.get('category', 'unknown')
                            category_counts[category] = category_counts.get(category, 0) + 1

                    # Compute the proportions
                    total_goals = len(self.goal_idxs)
                    category_info = []
                    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total_goals) * 100
                        category_info.append(f"{category}: {count}({percentage:.1f}%)")
                    
                    if 'tau' not in self.partition_kwargs:
                        raise ValueError("Partition strategy 'preference' requires 'omega' (or legacy 'tau') parameter in partition_kwargs.")
                    tau_value = self.partition_kwargs['tau']
                    print(f"[Federated WebShop TRAIN] Client {self.client_id}/{self.client_num} ({self.partition_strategy}, tau={tau_value}): "
                        f"Total: {partition_info['actual_samples']} goals, min: {min_goals_per_client}")
                    print(f"  Category distribution: {', '.join(category_info)}")
                elif self.partition_strategy == 'coverage':
                    # Show the parameter info for the coverage strategy
                    coverage_params = []
                    if 'size_std' in self.partition_kwargs:
                        coverage_params.append(f"size_std={self.partition_kwargs['size_std']}")
                    if 'overlap_ratio' in self.partition_kwargs:
                        coverage_params.append(f"overlap_ratio={self.partition_kwargs['overlap_ratio']}")
                    if 'min_samples_per_client' in self.partition_kwargs:
                        coverage_params.append(f"min_samples={self.partition_kwargs['min_samples_per_client']}")
                    if 'max_samples_per_client' in self.partition_kwargs:
                        coverage_params.append(f"max_samples={self.partition_kwargs['max_samples_per_client']}")
                    
                    param_str = f", {', '.join(coverage_params)}" if coverage_params else ""
                    print(f"[Federated WebShop TRAIN] Client {self.client_id}/{self.client_num} ({self.partition_strategy}{param_str}): "
                        f"Total: {partition_info['actual_samples']} goals, min: {min_goals_per_client}")
                elif self.partition_strategy == 'hardness':
                    # Show the parameter info for the hardness strategy
                    hardness_params = []
                    if 'success_std' in self.partition_kwargs:
                        hardness_params.append(f"success_std={self.partition_kwargs['success_std']}")
                    if 'trajectories_file' in self.partition_kwargs:
                        hardness_params.append(f"trajectories_file={self.partition_kwargs['trajectories_file']}")
                    
                    param_str = f", {', '.join(hardness_params)}" if hardness_params else ""
                    print(f"[Federated WebShop TRAIN] Client {self.client_id}/{self.client_num} ({self.partition_strategy}{param_str}): "
                        f"Total: {partition_info['actual_samples']} goals, min: {min_goals_per_client}")
                else:
                    print(f"[Federated WebShop TRAIN] Client {self.client_id}/{self.client_num} ({self.partition_strategy}): "
                        f"Total: {partition_info['actual_samples']} goals, min: {min_goals_per_client}")
            else:
                # Non-federated mode: use all training data
                total_train_goals = list(range(500, len(goals)))
                self.goal_idxs = total_train_goals
                print(f"[Standard WebShop TRAIN] Using all {len(total_train_goals)} training goals")

                print(f"Goal indices range: {min(self.goal_idxs)} - {max(self.goal_idxs)}")

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True


    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
    client_id: int = None,
    client_num: int = None,
    min_goals_per_client: int = 100,
    val_batch_size: int = 500,
    partition_strategy: str = 'uniform',
    infer_special: bool = False,
    start_idx: int = None,
    end_idx: int = None,
    **partition_kwargs
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
        client_id=client_id,  # Pass through client_id
        client_num=client_num,  # Pass through client_num
        min_goals_per_client=min_goals_per_client,  # Pass through min_goals_per_client
        val_batch_size=val_batch_size,  # Pass through val_batch_size
        partition_strategy=partition_strategy,  # Pass through partition_strategy
        infer_special=infer_special,  # Pass through infer_special
        start_idx=start_idx,  # Pass through start_idx
        end_idx=end_idx,  # Pass through end_idx
        **partition_kwargs  # Forward all partition-strategy parameters
    )