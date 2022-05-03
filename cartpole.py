import ray
ray.init()

import ray.rllib.agents.ppo as ppo

CHECKPOINT_ROOT = "tmp/ppo/cart"


SELECT_ENV = "CartPole-v1"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)

N_ITER = 5
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
  result = agent.train()
  file_name = agent.save(CHECKPOINT_ROOT)
  

  print(s.format(
    n,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"],
    file_name
   ))
  print(file_name)

print(file_name)

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"



