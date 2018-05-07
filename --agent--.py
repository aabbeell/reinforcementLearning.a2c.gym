import gym
import brain

class Agent:

    def __init__(self, env_name , lr_a, lr_c, gamma, epsilon):

        self.env = gym.make(env_name)

        self.action_space = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space
        self.render = False

        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.epsilon = epsilon
        

        self.networks = brain.A2C(self.lr_a, self.lr_c, self.gamma, self.epsilon, self.obs_dims, self.action_size)

    def act(self, state):
        return networks.get_action(state)

    def train(self, EPISODES, plot=False):
        
        scores, episodes = [], []
        
        for e in range(EPISODES):

            done = False
            state = self.env.reset()
            
            while not done: 

                if self.render = True:
                    self.env.render()

                action = act(state)
                next_state, reward, done, info = env.step(action)
                reward = reward if not done else -100   #or this is the last timestep in the episode
                networks.update(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                # every episode, plot the play time
                    score = score + 100
                    scores.append(score)
                    episode.append(e)
                    print("episode:", e, "  score:", score)

                if e%1000:
                    networks.save_weights()


        if plot=True:
            pass


