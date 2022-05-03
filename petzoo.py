import ray
import ray.rllib.agents.ppo as ppo

print("here")

ray.shutdown()
ray.init(ignore_reinit_error=True)
print("here")

print("Dashboard URL: http://{}".format(ray.get_webui_url()))

import shutil

CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

SELECT_ENV = "Taxi-v3"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)