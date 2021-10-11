**Overview**

Application of Reinforcement learning, Q-learning and neural networks in stock trading of 3 tech companies, Google, Amazon and Apple. The trading is short-term. Our model works upon closing prices in a window-size (in days) so as to determine the action to be taken at a specific time. 





**Results**

1. Best Window size for each stock amongst 5,10, 15 and 20 day window
![Profit vs Window size](https://github.com/ishatyagi22/stock-prediction-rl/blob/main/results/Profit%20vs%20Window%20Size.png)





**Running the Code**

Training the model : Retreive training and test data, csv files from  Yahoo! Finance into **`data/`**

```
python train AAPL 10 20
```

Evaluation after completion( 20 episodes) minimum of training : python evaluate.py 
```
python evaluate.py AAPL_2019 model_ep20
```


