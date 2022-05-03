import ray

ray.init()

from ray import tune

tune.run("PPO",
            config={'env' : "CartPole-v1",
                    "evaluation_interval" : 2, #num of training iter betn evaluations
                    "evaluation_num_episodes" : 20,
                    "num_gpus" : 1, 

            
            
            }


)