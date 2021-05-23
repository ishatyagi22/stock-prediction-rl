import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval
		self.firstIter = True

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model("models/" + model_name) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state):
		if(self.is_eval and self.firstIter):
			self.firstIter = False
			return 1

		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		
		options = self.model.predict(state)
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)

		# Confusing, read the replies after the post https://keon.github.io/deep-q-learning/
		# The real action of current state use the reward in sample (current reward + model predicted rewards in the future)
		# The opposite action use the rewards predicted by the model

		train_X = np.array([
			state[0] for state, action, reward, next_state, done in minibatch
		])

		# Only not done memories can use the predicted reward value
		not_done_mask = np.array([
			np.array([1, 1, 1]) * (np.arange(self.action_size) == action) *
			(not done) for state, action, reward, next_state, done in minibatch
		])
		done_mask = np.array([
			np.array([1, 1, 1]) * (np.arange(self.action_size) == action) * done
			for state, action, reward, next_state, done in minibatch
		])

		# Real rewards of current step
		rewards = np.array(
			[reward for state, action, reward, next_state, done in minibatch])
		# Next states after current actions
		next_states = np.array([
			next_state[0]
			for state, action, reward, next_state, done in minibatch
		])
		# Expected highest reward when using the next action, will replace the target reward
		target = rewards + (self.gamma *
							np.amax(self.model.predict(next_states), axis=1))
		target = np.repeat(target.astype("float32"), self.action_size).reshape(
			(-1, self.action_size))  # Make target the same size of target_f (y)
			
		train_y = self.model.predict(train_X)
		np.putmask(train_y, not_done_mask,
				   target)  # Replace with expected rewards
		np.putmask(
			train_y, done_mask,
			np.repeat(rewards.astype("float32"), self.action_size).reshape(
				(-1, self.action_size)))

		history = self.model.fit(train_X, train_y, epochs=1, verbose=0)
		loss = history.history['loss'][0]

		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

		return loss

