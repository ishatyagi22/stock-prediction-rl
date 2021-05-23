**Overview**

Application of Reinforcement learning, Q-learning and neural networks in stock trading of 3 tech companies, Google, Amazon and Apple. The trading is short-term. Our model works upon closing prices in a window-size (in days) so as to determine the action to be taken at a specific time. 





**Results**

1. Best Window size for each stock amongst 5,10, 15 and 20 day window
https://github.com/ishatyagi22/stock-prediction-rl/blob/main/results/Profit%20vs%20Window%20Size.png




2. Profit Without draw limit, sell and buy volume = 1 

#Apple :

#Google :

#Amazon : 


3. Profit with draw limit of $20000 and Volume ratio : 

#Apple :

#Google :

#Amazon : 


4. Profit with draw limit of $20000 and Volume ratio : 

#Apple :

#Google :

#Amazon : 


5. Profit with draw limit of $20000 and Volume ratio : 

#Apple :

#Google :

#Amazon : 



**Running the Code**

Training the model : Retreive training and test data, csv files from  Yahoo! Finance into **`data/`**

Evaluation after completion( 200 episodes minimum of training : python evaluate.py ^GSPC_2011 model_ep1000




