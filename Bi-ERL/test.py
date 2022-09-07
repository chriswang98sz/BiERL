import gym
env = gym.make('Pendulum-v1')
for i_episode in range(20):
	# 重置环境
    observation = env.reset()
    for t in range(100):
        print(observation)
        # 随机行为
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # 结束
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
