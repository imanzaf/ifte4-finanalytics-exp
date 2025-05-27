# ifte4-finanalytics-exp
experimental repo for Fin Analytics and ML at UCL

## Important Points (to include in presentation)
### Problem Statement (Iman)
- comparison of Q-learning vs Linear Models
- comparison of performance across different time periods
- comparison of introducing additional actions

- Does the model still perform well if a period of crisis or instability is used as the training?
    - Training: only crisis or only stable
    - Test: on both

- Problems
    - Action definition: buy/sell, but how much? 
    - Different market periods: stable, crisis, etc.
        - explain time periods
        - no transaction costs

### Choices


### Methodology (RL & Q-Learning)
- (Gabriella / Ariq) What RL is?
    - Markov Decision Process (MDP) framework
- (Gabriella / Ariq) Linear Model vs. Q-Learning
    - explorations vs exploitation trade-off
    - Q-learning is suitable in cases where the specific probabilities, rewards, and penalties are not completely known, as the agent traverses the environment repeatedly to learn the best strategy by itself. (1)
- (Rickey) Types of Q-Learning
- (Iman) Code / Model Implementation

### (Aadhira/Iman) Results
- Tables
- Visualisations
- results from different exploration rates? decay rates?


## Reference Links
1. https://neptune.ai/blog/markov-decision-process-in-reinforcement-learning


### Tasks
All - compare performance to market index 

Time periods
- Stable: Train (2017 - 2018) & Test (2019)
- Financial Crisis: Train (Oct 2007 to March 2009) & Test (April 2009 to Dec 2009)
- Covid: Train (2020 - 2022) & Test (2023)


1. RL Linear Model
    - Gabriella - stable period (define what stable means)
    - Ariq - crisis
    - Produce one results table: Returns from the model in test period.
                Stable      Crisis 1    Crisis 2
    Model 1
    Model 2
    S&P
    - Produce visualisation also comparing performance

2. RL Q Learning
    - Rickey - Financial crisis
    - Aadhira - covid
    - Iman - stable

                            Stable      Crisis 1    Crisis 2
    Q-learning (basic)
    Q-learning (DL)
    S&P
    - visualisations


### Environment / State
- S&P 500 stocks (top 6)
- Buy, Sell, Hold
- Quantity of each stock



## Stable Period Justification (2017 - 2019)

2017:
- One of the most stable and profitable years in market history
- S&P 500 gained about 19.4%
- Very low volatility (VIX averaged around 11)
- Steady economic growth
- No major market corrections

2018:
- Started strong but became more volatile
- Q4 2018 saw a significant correction (about 20% drop)
- Trade tensions with China began
- Interest rate concerns
- Overall still positive for the year but with more volatility

2019:
- Strong recovery from 2018 correction
- S&P 500 gained about 28.9%
- Some volatility from trade tensions
- Ended very strong

While 2017 was exceptionally stable, 2018-2019 had some notable volatility. However, compared to other periods, this could still be considered relatively stable because:
- The volatility was mostly contained to specific events
- The overall trend was positive
- There were no major economic crises
- The market showed resilience in recovering from corrections


## How does the Q-Learning Agent Work?
*explain code implementation*

1. Initialise a stock trading environment with the following features:
    - initial balance (amount invested)
    - historical stock data for period of interest
    - setting the initial state 
    (note: values are discretized as Q-learning works better with discrete values, otherwise table would be infinitely large)
        - discrete portfolio value (separated into buckets representing % of initial value)
        - discrete position value for each stock
        - 3 day moving-average of each stock (use momentum instead of daily price to reduce impact of random price movements -  more meaningful for agent to learn from)

2. Initialise Q-learning Agent with the following features:
    - set the required variables like state size, action size, learning and exploration rates, empty q-table

3. Train Agent
    - For each episode:
        - Reset environment to initial state
        - For each trading day:
            - Get current state (portfolio value, positions, momentum)
            - Choose action for each stock (hold/buy/sell + quantity level)
            - Execute trades based on actions
            - Calculate reward:
                - Base reward: percentage change in portfolio value
                - Penalties:
                    - Cash penalty (-0.0001): if holding more than 50% cash
                    - Transaction penalty (-0.00001 per trade): to discourage excessive trading
                    - Diversification penalty (-0.001): if invested in fewer than 2 stocks
                - Final reward = (portfolio change + penalties) * 100
            - Update Q-table with new knowledge
            - Move to next trading day
        - Track performance metrics (rewards, returns, portfolio value)
        - Save best performing Q-table

4. Test Agent
    - Load trained Q-table
    - Reset environment with test period data
    - For each trading day:
        - Get current state
        - Choose best action for each stock (no exploration)
        - Execute trades
        - Track portfolio value
    - Calculate final performance:
        - Compare returns against S&P 500 benchmark
        - Report final portfolio value and total return


### Additional details about the environment and any restrictions

- The implementation allows for stock selection inherently:
    - The agent can have 0% stake in some stocks
    - It only needs to maintain positions in at least 2 stocks
    - It can choose to hold (action_type = 0) for any stock
    - It can sell up to 100% of a position (quantity_level = 3)

- Short sales are currently not implemented.


# TODO 
- implement comparison of equal weighted portfolio for test
- implement comparison with buy and hold of S&P
- add visualisation


#### Explanations of Episodes

results from all episodes contribute to the agent's learning because:
Q-table Persistence:
The Q-table is maintained across all episodes
Each episode builds upon the knowledge from previous episodes
The Q-values are continuously updated based on new experiences
Learning Process:
Each episode starts fresh (reset environment)
But the Q-table retains all previous learning
The agent uses both:
Previous knowledge (old_value)
New experiences (reward + future value)
Update Formula Components:
(1 - learning_rate) * old_value: Retains previous knowledge
learning_rate * (reward + discount_factor * next_max): Incorporates new learning
learning_rate = 0.2 means 20% new learning, 80% previous knowledge
Exploration vs Exploitation:
Early episodes: High exploration (random actions)
Later episodes: More exploitation (using learned Q-values)
But always some exploration (min_exploration_rate = 0.02)
This is why the agent improves over time:
Early episodes: Mostly random actions, building initial Q-values
Middle episodes: Mix of exploration and exploitation, refining Q-values
Later episodes: Mostly exploitation with some exploration, fine-tuning strategy
