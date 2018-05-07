
import numpy as np
np.random.seed(1337)
from keras.layers import Dense, Input, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
#import matplotlib.pyplot as plt
import gym


class A2C:
    
    def __init__(self, lr_a, lr_c, decay, gamma, epsilon, obs_dims, action_size, batch_size=1):
        
        self.logging = False
        
        # hiperparams
        self.lr_a = lr_a                         #learning rate actor
        self.lr_c = lr_c                         #learning rate critic
        self.epsilon = epsilon                   #exploration rate ????????
        self.gamma = gamma                       #discount factor
        self.decay = decay
        self.batch_size = batch_size             #number of frames the agents gets after every action, or Exp. Replay
        
        # dimensions
        self.obs_dims = obs_dims                 #dimensions of the input
        self.obs_size = np.prod(obs_dims)        #size of the input
        self.action_size = action_size           #number of action
        print("NETWORK INIT, obsdim, obsize, actsize: ", self.obs_dims, self.obs_size, self.action_size)
        self.value_size = 1                      #output dim of the cirtic
        
        # models
        self.actor, self.critic = self.build_models()
        
        #self.actor = self.build_actor()
        #self.critic = self.build_critic()
        
        self.actor_optimizer, self.critic_optimizer = self.build_actor_optimizer(), self.build_critic_optimizer()
        
        # serialization
        self.load_checkpoint = False
        self.save_checkpoint = True
        self.save_destination = "/home/default/Desktop/AAbel_A2C/weights/"
        
        if self.load_checkpoint:
            self.actor.load_weights(self.save_destination +  str(self.epsilon) + "agent_weights_actor.h5")
            self.critic.load_weights(self.save_destination +  + str(self.epsilon) + "agent_weights_critic.h5")
        



        
    def build_models(self):
        
        observation = Input(batch_shape=(None, self.obs_size))
        
        # Shared Stream
        l1_shared = Dense(24,  activation='sigmoid', kernel_initializer='he_uniform')(observation)
        #l2_shared = Dense(8, activation='sigmoid', kernel_initializer='he_uniform')(l1_shared)

        # Actor Stream
        l3_actor = Dense(8, activation='sigmoid', kernel_initializer='he_uniform')(l1_shared)
        actor_output = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(l3_actor)

        # Critic Stream
        l3_critic= Dense(8, activation='sigmoid', kernel_initializer='he_uniform')(l1_shared)
        critic_output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(l3_critic)
        
        #model = Model(input=state_input, output=[actor, critic])

        actor = Model(inputs=observation, outputs=actor_output)
        critic = Model(inputs=observation, outputs=critic_output)

        optim_a = Adam(lr=self.lr_a, decay=(self.decay/2))
        #optim_c = Adam(lr=self.lr_c)

        # the loss function of policy network is : log(action_prob) * advantages , which is form of cross entropy.
        actor.compile(loss='categorical_crossentropy', optimizer=optim_a)
        #critic.compile(loss='mse', optimizer=optim_c)

        actor.summary()
        critic.summary()
        
        return actor, critic

    def build_actor_optimizer(self):
        
        #TODO: ? 
        
        action = K.placeholder(shape=( ))
        advantages = K.placeholder(shape=(None, self.action_size))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.00001 * entropy
        
        optimizer = Adam(lr=self.lr_a)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def build_critic_optimizer(self):
        
        target = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss_c = K.mean(K.square(target - value)) #mse
        #loss_c = K.mean(K.square(target))

        optimizer = Adam(lr=self.lr_c, decay=self.decay)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss_c)

        train = K.function([self.critic.input, target], [], updates=updates)

        return train
        
    
    
    def update(self, state, action, reward, next_state, done):

        # update policy network every episode
        
        target = np.zeros((1, self.value_size))          #target value to train the critic
        advantages = np.zeros((1, self.action_size))     #advantage of the action, to train the actor
        
        state = np.reshape(state, (1, self.obs_size))
        next_state = np.reshape(next_state, [1, self.obs_size])
        
        value = self.critic.predict(state)[0]            #value of current state
        next_value = self.critic.predict(next_state)[0]  #value of the next state
        

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.gamma * (next_value) - value   # TD learning:
            target[0][0] = reward + self.gamma * next_value  # target = Rt + gamma*V(St+1)
        
        
        
        if self.logging:
            
            print("state ", state)
            print("values: ", value, next_value)    
            print("adv: ", advantages)
            print("target :", target)
            print("---------------------")

        # or /w the optimizers
        #self.actor_optimizer([state, action, advantages])
        self.critic_optimizer([state, target])
        self.actor.fit(state, advantages, epochs=1, verbose=0)
        #self.critic.fit(state, target, epochs=1, verbose=0)
        
    
    def get_action(self, state):
        
        policy = self.actor.predict(np.reshape(state, [1, self.obs_size])).flatten()      #array with probs. of taking every action
        action = np.random.choice(self.action_size, 1, p=policy)[0]     #sampling from policys distribution
        if self.logging:             
            print("policy:", policy, " --> action:", action)
        return action
    
    
    def save_weights(self):
        if self.save_checkpoint:
            name_actor = self.save_destination + str(self.epsilon) + "agent_weights_actor.h5"
            name_critic = self.save_destination + str(self.epsilon) + "agent_weights_critic.h5"
            
            #close ur eyes---
            #file =  open(name_actor, "w+")
            #file.close()
            #file =  open(name_critic, "w+")
            #file.close()
            #---------------
            
            self.actor.save_weights(name_actor)
            self.critic.save_weights(name_critic)
        
#-------------------------------AGENT------------------------------

class Agent:

    def __init__(self, env_name , lr_a, lr_c, decay, gamma, epsilon):

        self.env = gym.make(env_name)
        self.scores = []

        self.action_size = self.env.action_space.n
        self.obs_dims = self.env.observation_space.shape[0]
        self.render = False

        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.epsilon = epsilon
        

        self.networks = A2C(self.lr_a, self.lr_c, decay, self.gamma, self.epsilon, self.obs_dims, self.action_size)

    def act(self, state):
        return self.networks.get_action(state)

    def train(self, EPISODES, plot=False):
        
        
        for e in range(EPISODES):

            done = False
            score = 0
            state = self.env.reset()
            
            while not done: 

                if self.render == True:
                    self.env.render()

                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                #reward = reward if not done else -100   #or this is the last timestep in the episode
                self.networks.update(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                # every episode, plot the play time
                    self.scores.append(score)
                    print("episode:", e, "  score:", score)

               


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

def save_scores(scores, filename, file_dir="/home/default/Desktop/AAbel_A2C/"):
    file_object  = open(file_dir + filename , "w+")
    #file_object.write("\n")
    #file_object.write(filename)
    #file_object.write("\n")
    for x in scores:
        file_object.write(str(x) + " ")
    file_object.close()


#-------------------------TRAIN/optimize -----------------------------
lr_short = [0.0016, 0.0032,0.0064]
lr_long = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032,0.0064, 0.009]
discount_rates = [0.98, 0.90]
decays = [0.0, 0.00001, 0.0001, 0.0005, 0.001, 0.005]
trainsteps = 1500
agents = []

""" #LEARNING RATE AND DISCOUNT FACTOR

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
"""
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
