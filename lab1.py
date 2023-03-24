import numpy as np
import random
import math
import gym
from gym import logger as gymlogger

gymlogger.set_level(40)  # error only


# We use this file to test, then transfer over to notebook

env = gym.make("CartPole-v1")

# Make bins, i.e make intervals between lower and upper
# then divide them into no_bins
def make_discrete(lower, upper, no_bins):
    return np.linspace(lower, upper, no_bins)

def generate_discrete_bins(no_bins=10, dims=4, no_actions=2):
    shape = [no_bins] * dims
    shape.append(no_actions)
    return np.zeros(shape=shape)

print(generate_discrete_bins().shape)

def generate_q_table(no_bins = 10):
    no_dbins = no_bins
    bins = np.array([
        make_discrete(-4.8, 4.8, no_dbins),
        make_discrete(-2.0, 2.0, no_dbins),  # original problem: -inf to inf
        make_discrete(-0.3, 0.3, no_dbins),  # -0.3 to 0.3 is good
        make_discrete(-2.0, 2.0, no_dbins),  # original problem: -inf to inf
    ])

    q = generate_discrete_bins(no_bins=no_dbins)

    return bins, q

def round_toNearestBin(observation, bins):
    pos, vel, angle, angular_vel = observation
    pos_bin = np.digitize(pos, bins[0])
    pos_vel = np.digitize(vel, bins[1])
    pos_angle = np.digitize(angle, bins[2])
    pos_angular_vel = np.digitize(angular_vel, bins[3])

    # its 1 based, need -1
    return pos_bin - 1, pos_vel - 1, pos_angle - 1, pos_angular_vel - 1

def learning_rate(episode_no: int, min_rate = 0.01) -> float:
    # need add 1 cos episode is 0 based
    return max(min_rate, min(1.0, 1.0 - math.log10((episode_no + 1) / 50)))

def discount_factor():
    return 0.995

def exploration_rate(episode_no: int, min_rate = 0.01) -> float:
    # need add 1 cos episode is 0 based
    return max(min_rate, min(1, 1.0 - math.log10((episode_no + 1) / 50)))

def policy(q, observation, episode_no):
    action = np.argmax(q[observation])

    if np.random.random() < exploration_rate(episode_no):
        action = random.randint(0, 1)

    return action

def q_learning_trainer(episodes = 5000,
                       no_bins = 10,
                       run_limit=120000,
                       terminate_on_percentage_score = (False, 0.5)):
    print(f"init trainer with, episode = {episodes}, no_bins = {no_bins}")
    
    bins, qt = generate_q_table(no_bins=no_bins)

    print(f"Table shape: {qt.shape}")

    episode = 0
    avg = 0
    fullscore_episodes_in_interval = 0
    interval_avg = 0

    while episode < episodes and episode < run_limit:
        observation_discrete = round_toNearestBin(env.reset(), bins)
        done = False
        reward_episode = 0

        # keep stepping t+1 into the future until done
        # which happens when it fails
        while not done:                
            action = policy(qt, observation_discrete, episode_no=episode)
            new_obs, new_reward, done, _ = env.step(action)
            future_obs = round_toNearestBin(new_obs, bins)

            max_future = np.max(qt[future_obs])
            q_current = qt[observation_discrete][action]

            q_New = q_current + learning_rate(episode_no=episode) * \
                ((new_reward + discount_factor() * max_future) - q_current)

            # update q
            qt[observation_discrete][action] = q_New

            # update previous obs to new one 
            observation_discrete = future_obs

            # track episode rewards
            reward_episode += new_reward
            
        avg = ((avg * (episode)) + reward_episode) / (episode + 1)

        interval_avg += reward_episode

        if reward_episode == 500.0:
            fullscore_episodes_in_interval += 1
        
        if terminate_on_percentage_score[0] and fullscore_episodes_in_interval / 1000 >= terminate_on_percentage_score[1]:
            print("Found 500/10000 full scores. Terminating training!")
            break

        # print logs
        # reset interval avg and fullscore in interval
        if episode % 1000 == 0:
            if episode != 0:
                interval_avg /= 1000
            print(f"Evaluating episode: {episode}, episode_reward: {reward_episode}, avg_so_far: {avg}, interval_avg: {interval_avg}, explore_r: {exploration_rate(episode)}, 500_in_interval: {fullscore_episodes_in_interval}")
            fullscore_episodes_in_interval = 0
            interval_avg = 0
        
        episode += 1
  
    return qt, bins

def q_learning_agent(qt, obs, bins):
    return np.argmax(qt[round_toNearestBin(obs, bins)])

def run_n_trained(qt, bins, render_to_screen = False, n = 1):
    res = np.zeros(n)
    for i in range(n):
        observation = env.reset()
        ep_reward = 0    
        done = False
        while not done:
            if render_to_screen:
                env.render()
            #your agent goes here
            action = q_learning_agent(qt, observation, bins) 
            observation, reward, done, _= env.step(action)
            ep_reward += reward

        res[i] = ep_reward

    if render_to_screen:
        env.close()
    
    return res

def main():
    q, bins = q_learning_trainer(episodes = 5000, no_bins=10)
    res = run_n_trained(q, bins, render_to_screen = False, n = 100)
    avg = np.average(res)
    
    print(f"Avg: {avg}")
    res = run_n_trained(q, bins, render_to_screen = True, n = 1)
    print(res)
main()