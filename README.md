# Prediction CLTV with BG-NBD and Gamma Gamma

### BG/NBD(Beta Geometric/Negative Binomial Distribution)(BUY TILL YOU DIE)
Modeling the distribution of the general population through probability and reduce it to the indivual.

### BG/NBD Model models two processes for Expected Number of Transaction:
    # Transaction Process(Buy)  + Dropout Process(Till You Die)
    # Transaction Process: The number of transactions that can be performed by a customer in a certain period of time, as long as he is alive, is distributed with the transaction rate parameter. 
    # That is, as long as a customer is alive, he will continue to make random purchases around his transaction rate
    # Transaction rates each varies specific to the customer, gamma is distributed for the entire audience (r, a)
    # DROPUT PROCESS(TILL YOU DIE): each customer has a dropout rate with probability p
    # Dropout rates vary for each customer and beta is distributed for the entire audience (a,b)


### GAMMA GAMMA MODEL
    # Used to estimate how much profit a customer can generate on average per transaction
    # We will find expected average profit.
    # The monetary value of a customer's transactions is randomly distributed around the average of the transaction values.
    # Average transaction value may vary between users over time, but does not vary for an individual user.
    # Expected transaction value is gamma distributed among all customers.

- recency: time since last purchase
- T: age of the customer. Weekly (how long before the date of analysis was the first purchase made)
- Frequency: total number of recurring purchases (freq>1)
- Monetary value: average earnings per purchase

### BUSINESS PROBLEM
    # The UK-based retail company wants to determine a roadmap for its sales and marketing activities. 
    # In order for the company to make medium-long term plans, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

### STEPS
- Data Preparation
- Data Analysis
- Preparation of Lifetime Data Structure
- Establishmnet of BG-NBD Model
- Establisment of Gamma Gamma Model
- Creating Customer Segment
