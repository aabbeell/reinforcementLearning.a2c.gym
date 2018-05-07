import numpy as np
np.random.seed(1337)
from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from skimage.color import rgb2gray
from skimage.transform import resize
import gym


class A2C:

    def __init__(self, lr_a, lr_c, gamma, epsilon, obs_dims, action_size, batch_size=1):

        self.logging = False

        # hiperparams
        self.lr_a = lr_a  # learning rate actor
        self.lr_c = lr_c  # learning rate critic
        self.epsilon = epsilon  # exploration rate ????????
        self.gamma = gamma  # discount factor
        #self.batch_size = batch_size  # number of frames the agents gets after every action, or Exp. Replay

        # dimensions
        self.obs_dims = obs_dims  # dimensions of the input
        self.obs_size = np.prod(obs_dims)  # size of the input
        self.action_size = action_size  # number of action
        print("NETWORK INIT, obsdim, obsize, actsize: ", self.obs_dims, self.obs_size, self.action_size)
        self.value_size = 1  # output dim of the cirtic

        # models
        self.actor, self.critic = self.build_models()

        # serialization
        self.load_checkpoint = False
        self.save_checkpoint = True
        self.save_destination = "/home/default/Desktop/AAbel_A2C/weights_atari/"
        
        if self.load_checkpoint:
            self.actor.load_weights(self.save_destination +  str(self.epsilon) + "agent_weights_actor.h5")
            self.critic.load_weights(self.save_destination +  + str(self.epsilon) + "agent_weights_critic.h5")


    def get_action(self, observation):
        policy = self.actor.predict(observation).flatten()  # array with probs. of taking every action
        action = np.random.choice(self.action_size, 1, p=policy)[0]  # sampling from policys distribution
        if self.logging:
            print("policy:", policy, " action: ", action)
        return action

    def build_models(self):
        print(self.obs_dims)
        input = Input(batch_shape=(1, 84, 84, 1))
        conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Flatten()(conv2)
        fc1 = Dense(64, activation='relu')(conv3)
        policy = Dense(self.action_size, activation='softmax')(fc1)
        value = Dense(1, activation='linear')(fc1)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.compile(optimizer=Adam(lr=self.lr_a), loss='categorical_crossentropy')
        critic.compile(optimizer=RMSprop(lr=self.lr_c,rho=0.99, epsilon=0.01), loss='mse')

        actor.summary()
        critic.summary()

        return actor, critic

    def update(self, state, action, reward, next_state, done):

        # update policy network every episode

        print(state.shape, next_state.shape)

        target = np.zeros((1, 1))  # target value to train the critic
        advantages = np.zeros((1, self.action_size))  # advantage of the action, to train the actor

        #state = np.reshape(state, (1, self.obs_dims))
        #next_state = np.reshape(next_state, [1, self.obs_dims])

        value = self.critic.predict(state)[0]  # value of current state
        next_value = self.critic.predict(next_state)[0]  # value of the next state

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.gamma * next_value - value  # TD learning:
            target[0][0] = reward + self.gamma * next_value  # target = Rt + gamma*V(St+1)

        if True:
            #print("state ", state)
            print("values: ", value, next_value)
            print("adv: ", advantages)
            print("target :", target)
            print("---------------------")

        # or /w the optimizers
        # self.actor_optimizer([state, action, advantages])
        # self.critic_optimizer([state, target])
        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


class Agent:

    def __init__(self, env_name, epsilon, gamma=0.99, lr_a=0.01, lr_c=0.01):

        self.env = gym.make(env_name)
        self.scores = []

        self.action_size = self.env.action_space.n
        self.state_dims = self.env.observation_space.shape
        self.obs_dims = (84, 84, 1)
        self.render = True

        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.epsilon = epsilon

        self.networks = A2C(self.lr_a, self.lr_c, self.gamma, self.epsilon, self.obs_dims, self.action_size)


    # preprocess the observation:
    # 1: convert to grayscale
    # 2: resize it to 84x84
    # 3: convert it to uint8 (*255 to not lose any pixel information)
    def preprocess(self, state):
        observation = rgb2gray(state)
        observation = resize(rgb2gray(observation), (self.obs_dims[0], self.obs_dims[1]), mode='constant')
        observation = np.uint8(observation*255)
        observation = np.reshape([observation], (1, 84, 84, 1))
        return observation


    # get the next action from the network
    def act(self, observation):
        return self.networks.get_action(observation)


    def train(self, episodes):

        for e in range(episodes):

            done = False
            score = 0
            state = self.env.reset()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            #for _ in range(np.random.randint(1, 30)):
            #    next_state, _, _, _ = self.env.step(1)
            #    state = next_state


            #single episode
            while not done:

                #to render or not to render?
                if self.render:
                    self.env.render()

                observation = self.preprocess(state)
                action = self.act(observation)

                next_state, reward, done, info = self.env.step(action)
                #next_state = self.preprocess(next_state)
                # reward = reward if not done else -100   #or this is the last timestep in the episode
                #self.networks.update(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    # every episode, plot the play time
                    self.scores.append(score)
                    print("episode:", e, "  score:", score)

        if e%1000==0:
            self.networks.save_weights()





#------------------------------PLOT-------------------------



def plot_MA(scores, ma=10, name=""):
    #plotting

    x, y = [], []
    maxes  = []
    temp = []
    moving_avg =[]
    m_x = []


    for i in range(len(scores)):
        temp.append(scores[i])
        m_x.append(i+1)
        if i % ma == 0:
        #    y.append(np.mean(temp))
            maxes.append(max(temp))
            temp = []
            x.append(i+1)
        if i < ma:
            moving_avg.append(scores[i])
        else:
            moving_avg.append(np.mean(scores[i-ma:i]))



    #y.append(scores[len(scores)-1])
    #x.append(i+1)

    #plt.plot(x, y)
    plt.plot(m_x, moving_avg, label = name)
    plt.scatter(x, maxes)
    plt.legend(loc='upper left')

    #plt.show()
#-------------------------SAVE SCORES------------------------
"""
lr_short = [0.0016, 0.0032,0.0064]
lr_long = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032,0.0064, 0.009]
discount_rates = [0.98, 0.90]
decays = [0.0, 0.00001, 0.0001, 0.0005, 0.001, 0.005]
trainsteps = 1500
agents = []

 #LEARNING RATE AND DISCOUNT FACTOR

for lr in lr_long:
    for dr in discount_rates:
        
        print("------------------------------ LEARNING RATE : ", lr, " ---- DISCOUNT RATE : ", dr, "----------------------------")
        
        ## same actor & critic lr
        new = Agent('CartPole-v1', lr, lr, dr, len(agents))
        name = str(len(agents))+ ".-" + str(lr) + "-" + str(lr) + "_" + str(dr)
        agents.append(new)
        new.train(trainsteps)
        #plot_MA(new.scores, name=name)
        save_scores(new.scores, name)
        ## different actor & critic lr
        new2 = Agent('CartPole-v1', lr/2, lr, dr, len(agents))
        agents.append(new2)
        new2.train(trainsteps)
        name = str(len(agents))+ ".-" + str(lr/2) + "-" + str(lr) + "_" + str(dr) + "(diff_lr)"
        save_scores(new2.scores, name)
        #plot_MA(new2.scores, name=name)

dr =  0.98
for lr in  lr_short:
    for decay in decays:
        print("------------------------------ LEARNING RATE : ", lr, " ---- decay : ", decay, "----------------------------")
        
        ## same actor & critic lr
        new = Agent('CartPole-v1', lr/2, lr,decay, dr, len(agents))
        name = "decay" + str(len(agents))+ ".-" + str(lr) + "-" + str(lr) + "_" + str(decay)
        agents.append(new)
        new.train(trainsteps)
        #plot_MA(new.scores, name=name)
        save_scores(new.scores, name)
        ## different actor & critic lr
        new2 = Agent('CartPole-v1', lr/2, lr,decay, dr, len(agents))
        agents.append(new2)
        new2.train(trainsteps)
        name = "decay" + str(len(agents))+ ".-" + str(lr/2) + "-" + str(lr) + "_" + str(decay) + "(second)"
        save_scores(new2.scores, name)
        #plot_MA(new2.scores, name=name)
        


#plt.show()
"""

SpaceNigger = Agent('SpaceInvaders-v4', 1)
SpaceNigger.train(20000)

save_scores(SpaceNigger.scores, "SpaceNigger_Score")