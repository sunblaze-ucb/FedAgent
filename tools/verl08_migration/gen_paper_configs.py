#!/usr/bin/env python
"""Emit the paper-faithful federated config matrix for the verl-0.8 overlay.

This mirrors the ORIGINAL FedAgent config tree 1:1 in STRUCTURE + NAMING (the family
layout config/{uniform,env_heterogeneity,task_heterogeneity,decentralized}/ and the
descriptive fed_<env>_<algo>_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-<strategy>_<knobs>
filenames). The file CONTENTS are verl-0.8 run_fed.py configs (flat schema), because that
is what `python -m fedagent.fed.run_fed --config <...>` consumes -- the migration changed
the runner, not the experiment design.

Coverage = the paper's 176 configs:
  uniform/<Model>/{main,main_seed1,main_seed2,centralized,local_client1-3}/{grpo,ppo}/   112
      4 backbones x 7 settings x 2 algos x 2 envs (webshop + alfworld). p-uniform.
  env_heterogeneity/<strategy>[_ppo]/                                                      16
      Qwen2.5-1.5B only, WebShop only (these perturb the catalog/search engine):
      catalog_split (grpo div 0/0.3/0.7/1.0; ppo div 1.0), bm25_reweighting (grpo N4,N8;
      ppo N4), field_subset_index (grpo N4,N8; ppo N4), lookalike_injection (grpo N2,N4;
      ppo N4), rank_wrapper (grpo N4; ppo N4).
  task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/                                         24
      Qwen2.5-1.5B only: preference(omega 0.01,0.99), coverage(std 1,256), hardness(success_std 1,256).
  decentralized/{ep_per_round_change,samples_change,selected_cl_change}/{grpo,ppo}/         24
      Qwen2.5-1.5B only, on the homog uniform baseline; each varies ONE protocol knob.
                                                                                   total = 176

Three migration fidelity fixes are baked in (see fedagent/docs/migration.md):
  (1) WEBSHOP_SEARCH_RETURN_N: env-het arms perturb the catalog/search and need the paper's
      top-K=200 to keep targets reachable; the uniform / task-het / decentralized / baseline
      WebShop runs use the engine default 50 (the original never raised it on those), so the
      non-het baselines match the 0.3.1 numbers. (run_fed default is 200; we set it explicitly.)
  (2) ALFWorld max_turns=50 (the env-spec config/envs/alfworld*.yaml carries it) -- the paper
      ran 50-turn episodes; a smaller cap can only lower ALFWorld success.
  (3) ALFWorld rollout.n=G (=8 by default; matches the WebShop group size) + a larger context
      window (max_model_len) sized for 50-turn transcripts.  GPU-VERIFY: confirm no OOM /
      prompt truncation at max_turns=50; raise max_model_len if episodes truncate.

NOTE on ppo_mini_batch_size: stock verl-0.8 multiplies it by rollout.n internally
(ray_trainer.py:1311), so the original's "1 update/rollout-batch" (GRPO 8 prompts x G=8 = 64
sequences; PPO 64 x G) is reproduced by ppo_mini_batch_size=8 (GRPO) / 64 (PPO) PROMPTS here.

Usage:
    python -m tools.verl08_migration.gen_paper_configs                 # all 176 -> fedagent/config/paper
    python -m tools.verl08_migration.gen_paper_configs --group-size 2  # cheap smoke (lower G)
"""
import argparse
import itertools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Federation protocol defaults (paper): N clients, M/round, T rounds, E epochs/round, min goals/client.
N, M, T, E, MIN_GOALS = 100, 2, 70, 3, 100

# uniform main table sweeps all four backbones; het/decentralized use a single backbone.
MODELS = [   # (dir name == HF model name, HF id used as model_path; offline users override)
    ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B-Instruct",   "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen2.5-7B-Instruct",   "Qwen/Qwen2.5-7B-Instruct"),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
]
HET_MODEL = ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")  # env/task/decentralized backbone

ENVS = ("webshop", "alfworld")
ALGOS = ("grpo", "ppo")

# uniform settings -> federation overrides. 3-seed replication varies base_seed (the original
# varied shuffle_seed 42/21/84); centralized/local use T=70xE=3 (=210 epochs WITH per-round goal
# re-draw) rather than the original rd-1/ep-210, because the verl-0.8 runner draws goal variety
# from ROUNDS (FEDAGENT_BASE_SEED threads the round) -- 1 round would repeat the same goals.
UNIFORM_SETTINGS = {
    "main":          dict(total=N, m=M, seed=42),
    "main_seed1":    dict(total=N, m=M, seed=21),
    "main_seed2":    dict(total=N, m=M, seed=84),
    "centralized":   dict(total=1, m=1, seed=42),                       # FedAvg of 1 client == continued centralized training
    "local_client1": dict(total=N, m=1, seed=42, local_client_id=21),  # paper "Local Agent Training" (uniform_single)
    "local_client2": dict(total=N, m=1, seed=42, local_client_id=42),
    "local_client3": dict(total=N, m=1, seed=42, local_client_id=84),
}

# env-het arms (WebShop only). (algo, orig_strategy_name, run_fed_partition, extra, p_suffix).
# orig_strategy_name -> the directory + filename token (mirrors the original); run_fed_partition
# -> the value run_fed.py actually consumes (bm25_reweighting->bm25_reweight etc.).
ENV_HET = [
    ("grpo", "catalog_split",      "catalog_split",     dict(env_div=0.0, keep_ratio=0.7), "catalog_split_div-0.0_keep-0.7"),
    ("grpo", "catalog_split",      "catalog_split",     dict(env_div=0.3, keep_ratio=0.7), "catalog_split_div-0.3_keep-0.7"),
    ("grpo", "catalog_split",      "catalog_split",     dict(env_div=0.7, keep_ratio=0.7), "catalog_split_div-0.7_keep-0.7"),
    ("grpo", "catalog_split",      "catalog_split",     dict(env_div=1.0, keep_ratio=0.7), "catalog_split_div-1.0_keep-0.7"),
    ("ppo",  "catalog_split",      "catalog_split",     dict(env_div=1.0, keep_ratio=0.7), "catalog_split_div-1.0_keep-0.7"),
    ("grpo", "bm25_reweighting",   "bm25_reweight",     dict(variant_n=4), "bm25_reweighting_N-4"),
    ("grpo", "bm25_reweighting",   "bm25_reweight",     dict(variant_n=8), "bm25_reweighting_N-8"),
    ("ppo",  "bm25_reweighting",   "bm25_reweight",     dict(variant_n=4), "bm25_reweighting_N-4"),
    ("grpo", "field_subset_index", "bm25_field_subset", dict(variant_n=4), "field_subset_index_N-4"),
    ("grpo", "field_subset_index", "bm25_field_subset", dict(variant_n=8), "field_subset_index_N-8"),
    ("ppo",  "field_subset_index", "bm25_field_subset", dict(variant_n=4), "field_subset_index_N-4"),
    ("grpo", "lookalike_injection","lookalike",         dict(variant_n=2), "lookalike_injection_N-2"),
    ("grpo", "lookalike_injection","lookalike",         dict(variant_n=4), "lookalike_injection_N-4"),
    ("ppo",  "lookalike_injection","lookalike",         dict(variant_n=4), "lookalike_injection_N-4"),
    ("grpo", "rank_wrapper",       "rank_wrapper",      dict(variant_n=4), "rank_wrapper_N-4"),
    ("ppo",  "rank_wrapper",       "rank_wrapper",      dict(variant_n=4), "rank_wrapper_N-4"),
]

# task-het arms (both envs). (run_fed_partition, extra, p_suffix). hardness needs a trajectories file.
TASK_HET = [
    ("preference", dict(omega=0.01), "preference_omega-0.01"),
    ("preference", dict(omega=0.99), "preference_omega-0.99"),
    ("coverage",   dict(size_std=1),   "coverage_std-1"),
    ("coverage",   dict(size_std=256), "coverage_std-256"),
    ("hardness",   dict(success_std=1),   "hardness_success_std-1"),
    ("hardness",   dict(success_std=256), "hardness_success_std-256"),
]

# decentralized protocol ablation (both envs), on the homog uniform baseline. family ->
# [(M, T, E, min_goals, p_suffix_knobs)] -- each varies ONE knob; p-uniform throughout.
DECENTRALIZED = {
    "ep_per_round_change": [(M, 210, 1, MIN_GOALS), (M, 42, 5, MIN_GOALS)],
    "samples_change":      [(M, T, E, 1000),        (M, T, E, 500)],
    "selected_cl_change":  [(1, T, E, MIN_GOALS),   (4, T, E, MIN_GOALS)],
}

_PORT = itertools.count(8300)   # distinct base port per config (concurrent runs never collide)


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"' if v == "" else v
    return str(v)


def fed_filename(env, algo, total, m, t, e, min_goals, p_suffix):
    return (f"fed_{env}_{algo}_total-{total}_cl-per-rd-{m}_rd-{t}_ep-per-cl-{e}"
            f"_min-goals-per-cl-{min_goals}_p-{p_suffix}")


def client_overrides(is_ppo, group_size, env_kind):
    """Rollout/optim Hydra overrides. WebShop keeps the verified shape (response 6144 GRPO /
    4096 PPO, max_model_len 8192, 15-turn episodes). ALFWorld widens the context window for
    50-turn transcripts (GPU-VERIFY) but keeps the same batch/group/critic recipe."""
    if env_kind == "alfworld":
        resp = 8192               # 50-turn budget; GPU-VERIFY (raise to 12288/16384 if truncated)
        max_model_len = 16384     # holds the running 50-turn chat transcript; GPU-VERIFY for OOM
        gpu_mem = "0.5" if is_ppo else "0.6"
    else:
        resp = 4096 if is_ppo else 6144
        max_model_len = 8192
        gpu_mem = "0.5" if is_ppo else "0.6"
    batch = 64 if is_ppo else 8
    mini = 64 if is_ppo else 8
    ov = [
        f"data.train_batch_size={batch}",
        "data.max_prompt_length=2048",
        f"data.max_response_length={resp}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini}",   # PROMPTS; verl-0.8 x rollout.n => full batch = 1 update (== FedAgent)
        f"actor_rollout_ref.rollout.n={group_size}",             # FedAgent group size G (env.rollout.n=8)
        "actor_rollout_ref.rollout.prompt_length=2048",
        f"actor_rollout_ref.rollout.response_length={resp}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem}",
    ]
    if is_ppo:
        ov += [
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.actor.use_kl_loss=true",
            "actor_rollout_ref.actor.kl_loss_coef=0.01",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
            "actor_rollout_ref.actor.checkpoint.save_contents=[model]",
            "critic.optim.lr=1e-5",
            "critic.model.use_remove_padding=true",
            "critic.model.enable_gradient_checkpointing=true",
            "critic.fsdp.optimizer_offload=true",   # verl 0.8: critic FSDP at critic.fsdp
            "critic.ppo_micro_batch_size_per_gpu=4",
            "critic.checkpoint.save_contents=[model]",
            "trainer.critic_warmup=0",
        ]
    return ov


def env_specs(env_kind, is_ppo):
    if env_kind == "alfworld":
        return "config/envs/alfworld.yaml", "config/envs/alfworld_val.yaml"
    return (f"config/envs/webshop_15{'_ppo' if is_ppo else ''}.yaml",
            "config/envs/webshop_15_val.yaml")


def search_return_n(env_kind, family):
    """WebShop BM25 top-K. env-het perturbs the catalog/search -> needs 200 to keep targets
    reachable; everything else uses the engine default 50 (matches the original baselines).
    None for ALFWorld (search_return_n is WebShop-only)."""
    if env_kind != "webshop":
        return None
    return 200 if family == "env_heterogeneity" else 50


def build_config(header, *, env_kind, algo, model, total, m, t, e, seed, min_goals,
                 partition, extra, family, local_client_id=None):
    is_ppo = (algo == "ppo")
    env_spec, val_spec = env_specs(env_kind, is_ppo)
    sn = search_return_n(env_kind, family)
    port = next(_PORT)
    lines = [
        f"# {header}",
        "# auto-generated by tools/verl08_migration/gen_paper_configs.py -- mirrors the original",
        "# config/ tree (structure + naming); contents are verl-0.8 run_fed.py configs.",
        "",
        f"env_kind: {env_kind}",
        f"env_spec: {env_spec}",
        f"val_env_spec: {val_spec}",
        f"output_dir: /tmp/xbb9020_fedpaper/{family}/{env_kind}_{algo}_{partition or 'uniform'}",
        f"model_path: {model}                  # HF id; offline clusters: --model-path <local snapshot>",
    ]
    if is_ppo:
        lines += ["", "adv_estimator: gae                                  # PPO: federate actor + critic"]
    lines += [
        "",
        "# --- federation protocol ---",
        f"total_clients: {total}",
        f"clients_per_round: {m}",
        f"total_rounds: {t}",
        f"epochs_per_round: {e}",
        f"base_seed: {seed}",
    ]
    if local_client_id is not None:
        lines.append(f"local_client_id: {local_client_id}             # Local Agent Training: pin this client, no federation")
    lines += [
        "",
        "# --- 4-GPU FSDP recipe + unperturbed validation ---",
        "n_gpus_per_node: 4",
        "total_training_steps: 0                             # 0 => full E epochs/round (no per-round step cap)",
        "save_freq: 100000                                   # save only the round's last step",
        "test_freq: 5",
        "val_before_train: true",
        "val_temperature: 0.4",
        "wait_between_clients: 8",
        f"min_goals_per_client: {min_goals}",
    ]
    if env_kind == "webshop":
        lines += ["webshop_pool_size: 16", f"webshop_base_port: {port}"]
        lines.append(f"search_return_n: {sn}                               "
                     + ("# env-het perturbs the catalog -> paper top-K" if sn == 200
                        else "# engine default (matches the original non-het baselines)"))
    else:
        lines += ["alfworld_pool_size: 8", f"alfworld_base_port: {8400 + (port - 8300)}"]
    lines += ["", "# --- heterogeneity ---", f"partition_strategy: {_fmt(partition)}"]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {_fmt(v)}")
    if partition == "hardness":
        tag = "qwen2.5-1.5b"   # het backbone
        traj = f"data/hardness/{tag}_{env_kind}_trajectories.json"
        if env_kind == "webshop":
            # WebShop reference-policy labels are produced by gen_hardness_trajectories.py and
            # ship in data/hardness/ (qwen2.5-1.5b); see data/hardness/README.md for coverage.
            note = ("# reference-policy labels (see data/hardness/README.md); "
                    "regenerate via tools/verl08_migration/gen_hardness_trajectories.py")
        else:
            # ALFWorld labels need a SEPARATE reference pass -- the generator is WebShop-only.
            note = ("# REQUIRED, NOT shipped: ALFWorld needs a separate reference pass "
                    "(generator is WebShop-only) -- see data/hardness/README.md")
        lines.append(f"trajectories_file: {traj}   {note}")
    lines += ["", "client_overrides:"]
    lines += [f"  - {o}" for o in client_overrides(is_ppo, GROUP_SIZE, env_kind)]
    return "\n".join(lines) + "\n"


def emit_uniform(out):
    n = 0
    for (mdir, mid), (sname, s), algo, env in itertools.product(
            MODELS, UNIFORM_SETTINGS.items(), ALGOS, ENVS):
        total, m, seed = s["total"], s["m"], s["seed"]
        lid = s.get("local_client_id")
        d = out / "uniform" / mdir / sname / algo
        d.mkdir(parents=True, exist_ok=True)
        fn = fed_filename(env, algo, total, m, T, E, MIN_GOALS, "uniform")
        text = build_config(
            f"UNIFORM {sname} | {algo.upper()} | {env} | {mdir}",
            env_kind=env, algo=algo, model=mid, total=total, m=m, t=T, e=E, seed=seed,
            min_goals=MIN_GOALS, partition="", extra={}, family="uniform", local_client_id=lid)
        (d / f"{fn}.yaml").write_text(text)
        n += 1
    return n


def emit_env_het(out):
    mdir, mid = HET_MODEL
    n = 0
    for algo, orig, part, extra, psuf in ENV_HET:
        d = out / "env_heterogeneity" / (orig + ("_ppo" if algo == "ppo" else ""))
        d.mkdir(parents=True, exist_ok=True)
        fn = fed_filename("webshop", algo, N, M, T, E, MIN_GOALS, psuf)
        text = build_config(
            f"ENV-HET {orig} | {algo.upper()} | webshop | {mdir}",
            env_kind="webshop", algo=algo, model=mid, total=N, m=M, t=T, e=E, seed=42,
            min_goals=MIN_GOALS, partition=part, extra=extra, family="env_heterogeneity")
        (d / f"{fn}.yaml").write_text(text)
        n += 1
    return n


def emit_task_het(out):
    mdir, mid = HET_MODEL
    n = 0
    for algo, env, (part, extra, psuf) in itertools.product(ALGOS, ENVS, TASK_HET):
        d = out / "task_heterogeneity" / algo / env
        d.mkdir(parents=True, exist_ok=True)
        fn = fed_filename(env, algo, N, M, T, E, MIN_GOALS, psuf)
        text = build_config(
            f"TASK-HET {part} | {algo.upper()} | {env} | {mdir}",
            env_kind=env, algo=algo, model=mid, total=N, m=M, t=T, e=E, seed=42,
            min_goals=MIN_GOALS, partition=part, extra=extra, family="task_heterogeneity")
        (d / f"{fn}.yaml").write_text(text)
        n += 1
    return n


def emit_decentralized(out):
    mdir, mid = HET_MODEL
    n = 0
    for family, env, algo in itertools.product(DECENTRALIZED, ENVS, ALGOS):
        d = out / "decentralized" / family / algo
        d.mkdir(parents=True, exist_ok=True)
        for (m, t, e, min_goals) in DECENTRALIZED[family]:
            fn = fed_filename(env, algo, N, m, t, e, min_goals, "uniform")
            text = build_config(
                f"DECENTRALIZED {family} (M={m},T={t},E={e},min_goals={min_goals}) | {algo.upper()} | {env} | {mdir}",
                env_kind=env, algo=algo, model=mid, total=N, m=m, t=t, e=e, seed=42,
                min_goals=min_goals, partition="", extra={}, family="decentralized")
            (d / f"{fn}.yaml").write_text(text)
            n += 1
    return n


def main():
    global GROUP_SIZE
    ap = argparse.ArgumentParser(description="emit the paper-faithful 176-config matrix (mirrors original config/)")
    ap.add_argument("--out", default=str(REPO_ROOT / "fedagent" / "config" / "paper"))
    ap.add_argument("--group-size", type=int, default=8,
                    help="rollout.n = GRPO group size G (FedAgent default 8; lower for a cheap smoke)")
    args = ap.parse_args()
    GROUP_SIZE = args.group_size

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {
        "uniform": emit_uniform(out),
        "env_heterogeneity": emit_env_het(out),
        "task_heterogeneity": emit_task_het(out),
        "decentralized": emit_decentralized(out),
    }
    total = sum(counts.values())
    print(f"wrote {total} configs to {out}:")
    for k, v in counts.items():
        print(f"  {k:20s} {v}")
    assert total == 176, f"expected 176 paper configs, got {total}"
    print("\n3-seed replication: base_seed 42/21/84 are uniform/main, main_seed1, main_seed2.")
    print("hardness arms: generate trajectories_file via gen_hardness_trajectories.py before running.")
    print("ALFWorld max_turns=50 + widened context are GPU-VERIFY (see docstring fix #2/#3).")


if __name__ == "__main__":
    main()
