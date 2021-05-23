import keras
from keras.models import load_model

import math
from agent.agent import Agent
from functions import *
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
buy_points = []
sell_points = []
total_cash = 20000
initial_cash = total_cash
min_cash_reserve = total_cash
do_single_cell = False
volume_ratio = 0.25

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1 and total_cash >= data[t]: # buy
		num_to_buy = max(math.ceil((total_cash / data[t]) * volume_ratio), 1)
		agent.inventory.extend([data[t]]*num_to_buy)
		print("Buy: " + formatPrice(data[t]), "| # Shares: ", num_to_buy)
		total_cash -= data[t]*num_to_buy
		min_cash_reserve = min(min_cash_reserve, total_cash)
		buy_points.append([t, data[t], num_to_buy])

	elif action == 2 and len(agent.inventory) > 0: # sell
		current_inventory = len(agent.inventory)
		num_to_sell = 1 if do_single_cell else math.ceil(current_inventory * volume_ratio)
		to_sell, agent.inventory = agent.inventory[:num_to_sell], agent.inventory[num_to_sell:]
		bought_price = sum(to_sell)
		sell_price = data[t]*num_to_sell
		reward = max(data[t] - (bought_price / num_to_sell), 0)
		total_profit += sell_price - bought_price
		sell_points.append([t, data[t], num_to_sell])
		total_cash += sell_price
		print("Sell: " + formatPrice(data[t]) + " | # Shares: " + str(num_to_sell) + " | Profit: " + formatPrice(sell_price - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print("--------------------------------")
		print(stock_name + " Total Profit: " + formatPrice(total_profit))
		print("Max Draw: " + str(initial_cash - min_cash_reserve))
		print("--------------------------------")
		result_dir = "results/" + stock_name + "/" + str(window_size) + "/" + model_name.split("/")[0] + "/"
		if not os.path.isdir(result_dir):
			os.makedirs(result_dir)
		figure = "profit:{}.png".format(round(total_profit,0))
		plt.figure(figsize = (20, 10))
		plt.plot(data, label = stock_name, c = 'black')
		buy_points = np.array(buy_points)
		sell_points = np.array(sell_points)

		#plt.plot(data, 'o', label = 'buy', markevery = buy_points[:, 0].astype(int).tolist(), c = 'g')
		#plt.plot(data, 'o', label = 'sell', markevery = sell_points[:, 0].astype(int).tolist(), c = 'r')
		
		plt.scatter(buy_points[:,0], buy_points[:, 1], s=buy_points[:, 2]*5, label='buy', c = 'g', marker='o', alpha=1, edgecolors='face', zorder=5)
		plt.scatter(sell_points[:,0], sell_points[:, 1], s=sell_points[:, 2]*5, label='sell', c = 'r', marker='o', alpha=1, edgecolors='face', zorder=5)
		plt.legend()
		plt.savefig(result_dir + figure) 