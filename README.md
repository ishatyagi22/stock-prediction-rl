**Overview**

Application of Reinforcement learning, Q-learning and neural networks in stock trading of 3 tech companies, Google, Amazon and Apple. The trading is short-term. Our model works upon closing prices in a window-size (in days) so as to determine the action to be taken at a specific time. 





**Results**

1. Best Window size for each stock amongst 5,10, 15 and 20 day window
![Profit vs Window size](https://github.com/ishatyagi22/stock-prediction-rl/blob/main/results/Profit%20vs%20Window%20Size.png)



2. Profits with different draw limit and buying and sell volume ratio. 
   Buying ratio is fraction of cash balance that can be used to BUY stock. 
   Sell ratio is fraction of stock inventory that can sold.

   1. Profit Without draw limit, sell and buy volume = _maximum 1 stock per day_

      - Apple : Profit: $418.0 ![](https://github.com/ishatyagi22/stock-prediction-rl/blob/main/results/AAPL_2019/10/AAPL/profit:418.0.png)

      - Google :

      - Amazon : 


   2. Profit with draw limit of $20000 and Volume = _maximum 1 stock per day_

      - Apple :

      - Google :

      - Amazon : Profit: $1324.0 ![Amazon](https://github.com/ishatyagi22/stock-prediction-rl/blob/main/results/AMZN_2019/20/AMZN/profit:1324.0.png)


   3. Profit with draw limit of $20000 and Volume ratio = 0.25

      - Apple :

      - Google :

      - Amazon : 


   4. Profit with draw limit of $20000 and Volume ratio : 0.5

      - Apple :

      - Google :

      - Amazon : 



**Running the Code**

Training the model : Retreive training and test data, csv files from  Yahoo! Finance into **`data/`**

```
python train AAPL 10 20
```

Evaluation after completion( 20 episodes) minimum of training : python evaluate.py 
```
python evaluate.py AAPL_2019 model_ep20
```


