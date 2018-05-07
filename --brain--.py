class A2C:
    
    def __init__(self, lr_a, lr_c, gamma, epsilon, obs_dims, action_size, batch_size=1):
        
        
        # hiperparams
        self.lr_a = lr_a                         #learning rate actor
        self.lr_c = lr_c                         #learning rate critic
        self.epsilon = epsilon                   #exploration rate ????????
        self.gamma = gamma                       #discount factor
        self.batch_size = batch_size             #number of frames the agents gets after every action, or Exp. Replay
        
        # dimensions
        self.obs_dims = obs_dims                 #dimensions of the input
        self.obs_size = np.prod(obs_dims)        #size of the input
        self.action_size = action_size           #number of action
        self.value_size = 1                      #output dim of the cirtic
        
        # models
        self.actor, self.critic = self.build_models()
        #self.actor_optimizer, self.critic_optimizer = self.build_actor_optimizer(), self.build_critic_optimizer()
        
        # serialization
        self.load_checkpoint = False
        self.save_checkpoint = False
        self.save_destination = "/Users/daddy/Desktop/untitled folder/"
        
        if self.load_checkpoint:
            self.actor.load_weights(self.save_destination + "weights_actor.h5")
            self.critic.load_weights(self.save_destination + "weights_critic.h5")
        
        
    def build_models(self):
        
        observation = Input(batch_shape=(None, self.obs_size))
        
        # Shared Stream
        layer1 = Dense(32,  activation='relu', kernel_initializer='he_uniform')(observation)
        layer2 = Dense(16, activation='relu', kernel_initializer='he_uniform')(layer1)

        # Actor Stream
        actor_output = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(layer2)

        # Critic Stream
        critic_output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(layer2)
        
        #model = Model(input=state_input, output=[actor, critic])

        actor = Model(inputs=observation, outputs=actor_output)
        critic = Model(inputs=observation, outputs=critic_output)

        adam_a = Adam(lr=self.lr_a, clipnorm=1.0)
        adam_c = Adam(lr=self.lr_c, clipnorm=1.0)

        # the loss function of policy network is : log(action_prob) * advantages , which is form of cross entropy.
        actor.compile(loss='categorical_crossentropy', optimizer=adam_a)
        critic.compile(loss='mse', optimizer=adam_c)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()
        
        return actor, critic
        
    #TODO    
    def build_actor_optimizer(self):
        pass
        actions = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, self.action_size))
        
        policy = self.actor.actor_output

        """
        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)
        -----
        
        action_prob = K.sum(action * policy, axis = 1)
        loss_a = -K.sum(K.log(action_prob + 1e-10) * advantages)

        """
        
        # adding entropy to the actor's loss function to encourage exploration in the early phase

        

        entropy = -K.sum(self.actions * K.log(self.actions))
        losses = -(K.log(actions) * advantages + 0.01 * .entropy)
        loss_a = K.reduce_sum(losses)

        optimizer = Adam(lr=self.lr_a)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss_a)
        train = K.function([self.actor.input, actions, advantages], [], updates=updates)

        return actor_optimizer
    

    #TODO
    def build_critic_optimizer(self):
        pass
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss_c = K.mean(K.square(discounted_reward - value)) #mse

        optimizer = Adam(lr=self.lr_c)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss_c)

        train = K.function([self.critic.input, discounted_reward], [], updates=updates)

        return train
        
    
    
    def update(self, state, action, reward, next_state, done):

        # update policy network every episode
        target = np.zeros((1, self.value_size))          #target value to train the critic
        advantages = np.zeros((1, self.action_size))     #advantage of the action, to train the actor

        value = self.critic.predict(state)[0]            #value of current state
        next_value = self.critic.predict(next_state)[0]  #value of the next state

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value   # TD learning:
            target[0][0] = reward + self.discount_factor * next_value                      # = Rt + gamma*V(St+1)
            
            
        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

        # or /w the optimizers
        #self.optimizer[0]([state, action, advantages])
        #self.optimizer[1]([state, target])
        
    
    def get_action(self, state):
        
        policy = self.actor.predict(state, batch_size=1).flatten()      #array with probs. of taking every action
        print(policy)
        action = np.random.choice(self.action_size, 1, p=policy)[0]     #sampling from policys distribution
        return action
    
    
    def save_weights(self):
        self.actor.save_weights(self.save_destination + "weights_actor.h5")
        self.critic.save_weights(self.save_destination + "weights_critic.h5")
        