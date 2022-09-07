from gym import envs
env_names = [spec.id for spec in envs.registry.all()] 
for name in sorted(env_names): 
    print(name)
import gym
env = gym.make('Qbert-ram-v4')
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs)
env.close()
 
