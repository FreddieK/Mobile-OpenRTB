# Mobile OpenRTB
Before leaving mobile Adtech for the second time, writing down some notes for possible future reference.

- [Mobile OpenRTB](#mobile-openrtb)
  - [Fundamentals](#fundamentals)
  - [Impression Value](#impression-value)
    - [Imbalanced Learning](#imbalanced-learning)
    - [Non-stationary Landscape and Concept Drift](#non-stationary-landscape-and-concept-drift)
  - [Bid Shading](#bid-shading)
    - [Market Price Estimation](#market-price-estimation)
    - [Alterntive Approaches to Bid Shading](#alterntive-approaches-to-bid-shading)
  - [Opportunity Cost](#opportunity-cost)
    - [Temporal Budget Planning](#temporal-budget-planning)
    - [Pacing](#pacing)
  - [Additional Challenges](#additional-challenges)
    - [Explore-Exploit Tradeoff](#explore-exploit-tradeoff)
    - [Feedback Loops](#feedback-loops)
    - [Handling Margin](#handling-margin)
    - [Experimentation and Evaluation](#experimentation-and-evaluation)

## Fundamentals
[Rådström (2018)](https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8953440&fileOId=8953444) provides an impressive overview in his master's thesis, covering the main high-level challenges; 
* Impression Value / Action rate predictions
* Bid Shading / Profit optimization
* Opportunity Cost / Temporal budget allocation

Before going down various rabbit holes, this still serves as a great introduction that takes time to grasp.

## Impression Value
Impression value is most simply ```p(action) * valueOfAction```. Going a bit more sophisticated, you might consider whether the action is incremental or not, as well as break out the attribution mechanism, since we're typically relying on a third party attribution provider. 

```p(action|impression)*p(attribution|impression) - p(action)*p(attribution)```

Each of these predictions can have more or less sophistication. Did we have an impression with click on the user an hour ago, but the user didn't install? How likely is it that the user will install soon, and how it that likelihood impacted by an additional impression? How likely is it that a competitor has gotten a click for the user since then and stolen the attribution window?

### Imbalanced Learning
One big challenge here is that when you go beyond click-based predictions to instead predict events such as install or purchase, you end up with extreme data sparsity for positive labels.

This is one reason why neural network based models often are preferrable to tree based models, as various pre-training and/or multi-modal techniques can be utilised to generate embeddings for categorical variables. Another related reason is the high cardinality of categorical variables, where for example publishers usually number in the thousands, and Android device models in the hundreds.

Finding ways to improve signal strength, and reduce the issue of cold-start for new advertisers, is a core focus.

This can cause problems with model convergence due to lack of positive samples in each batch during training, and techniques like large batch size and over/under sampling can become important (while we still in the end want to recalibrate predictions). In addition, given typically huge data sizes, downsampling is an effective way to improve training times.

**Pitfalls of predictions at the extremes**

The Sigmoid function that transforms the final score from the model into a prediction is practically linear in impact for predictions in the range 10-90%, i.e. for more balanced data sets. At the extremes though, the impact of a change in input score has exponential impact on the predicted probabilities, which easily causes uneven calibration in different prediction ranges.

In combination with us using the predicted probability score, rather than finding a binary classification threshold, this can lead to issues with bidding too aggressively in the upper ranges due to calibration issues.

This is further exasperated if logloss is used in the loss function, where more certain predictions are punished more harshly, which can lead them to dominate the learning and put lower emphasis on the higher predictions, which in fact are the ones we would rather want to emphasize during learning.

Together, these aspects further motivates the use of techniques such as negative downsampling to help the model learn more evenly calibrated predictions, though they then in the end need to be recalibrated based on the original base rate.

### Non-stationary Landscape and Concept Drift
The challenge of sparsity of positive samples can be reduced by increasing the time window used for training. This comes with the tradeoff though that the OpenRTB landscape is non-stationary, with publishers and device models coming and going, and SSPs launching new SDK versions and a host of other thing changing.

Thus, balancing use of historic data vs. newer, and ensuring the model is robust when predicting out of distribution compared to how it was trained is important to ensure the online performance doesn't degrade significantly compared to offline.

## Bid Shading
Historically, openRTB auctions were mainly second price sealed auctions, at least in theory. As the advertisers have no insight into the supply side auction, the mechanism was built on trust, and the SSPs had incentive to optimize their earnings which led to the introduction of various yield optimization mechanisms such as 'soft' floor prices and waterfall setups.

This practically meant bidding your value was a losing strategy, and from the advertiser perspective, treating everything as a first price sealed auction was a safer option.

Given the introduction of header bidding, where SSPs in addition were competing against each other to return the highest won bid to the publisher, everyone benefited from simplifying the setup and use first price auctions instead, which the eco system gradually has migrated to.

This leads to a simple formula for optimizing bids in a single auction as;

```
utility = (impressionValue - bid)*winrate(bid)

argmax(utility|bid)
```

We select the bid that maximises the expected utility.

This formula will show up, using various notation, in most papers covering openRTB advertising, and approaches to bid shading typically either predicts the market price of inventory, or tries to optimise the de facto utility outcome of historic bids, i.e.;

```
(impressionValue - bid) if impression == 1 else 0
```

One extension here is to also include an infrastructure cost to making the bid, in order to encourage not bidding if the expected utility is too low (typically due to winrate going towards zero).

Similar to the impression value, concept drift is also a problem when it comes to bid shading.

### Market Price Estimation
A common approach is to model the market price distribution for different inventory segments, either parametrically or non-parametrically. If parametrically, integrating over the PDF give the CDF where we easily can predict the win rate for any bid. If non-parametric, model learns to predict win rate at a discrete number of bid candidates.

A core challenge for market price prediction is that we typically only have censored binary feedback, whether we won or not, but not what we would have needed to bid in order to win.

There are signals such as `min_bid_to_win` provided by some SSPs, but the signal is not standardised in the interpretation, and it might just mean we won the RTB auction, but still lost in a subsequent mediation layer step.

Further, there is a more fundamental issue with the utility formula in predicting the market price in that modelling the problem as a sealed first price auction is a very poor match for the reality of the OpenRTB advertising system, meaning that the game theoretically optimal approach for sealed first price auctions have no guarantee of being optimal in the actual game we're playing.

### Alterntive Approaches to Bid Shading
A very fundamental issue with the openRTB landscape is that though for convenience it's often modelled as a sealed first or second price auction, in reality it is much more complex and is more akin to a hybrid between a Dutsch falling price auction and a sealed auction.

The **Waterfall Setup** of publishers lead to multi-round auctions with soft floor prices, where if we don't participate in one round we might still receive an opportunity to bid on the inventory again with a lower floor price. At each round though, we don't know what the true reserve price might be, nor how this bid request relates to previous or later bid requests. 

This also makes it important for the bid shading mechanism to be able to lead to not bidding, and not only bid shade in the range ```floor price <= bid <= impressionValue```.

**Header Bidding** allow publishers to work with multiple SSPs in parallel, meaning the DSP might receive two or more bid requests in parallel for the same impression opportunity, meaning that besides bidding against competitors, we might be bidding against ourselves.

Due to these aspects, and the censored learning making market price prediction hard, many alternative approaches to bid shading exists, including survival modelling, hill climbing (assuming a convex but non-differentiable landscape) and reinforcement learning (either while still modelling the impression value, or using end-to-end reinforcement learning).

## Opportunity Cost
Since we're having a limited ad budget distributed over a high number of fungible items in repeated auctions, we can't treat each auction in isolation but must also factor in possible opportunity cost on spending budget now vs. saving it for later.

This is typically done via a control signal from a plant that relatively shrinks the impression value. When we are pacing faster than desired, we impose a higher required minimum return on ad spend (ROAS).

```
impressionValue = impressionValue/minROAS
```

This will lead to us participating in fewer auctions due to the impression value ending up lower than the floor price, as well as incentivise lower bids in the utility formula.

### Temporal Budget Planning
We can't know for sure what the future look like, but typically we have some constraints with a budget needing to be spent within some timeframe, and likely with some degree of even distribution (~spend per day).

Within these constraints, we want to device a plan for how the budget should be spent, based on what we believe will generate the best returns.

Typically in papers, this is formulated as a mathematical optimization problem, and then reformulated as a Lagrangian function that you try to solve.

Besides using these aspects to break down spend in time, the same type of planning algorithm can also be extended to subdivide the spend across different campaigns etc, if the more tactical impression value estimate end up not being robust enough in aggregate.

Three important aspects for how to distribute budget in time;

**Inventory availability**

On a 24-hour daily cycle, we have very typical patterns with low traffic during the night, gradually rising during morning and a first peak at lunch, after which a slowdown until late afternoon when it starts rising again until the evening daily peak.

Similarly, Sundays are typically the day of the week when most traffic is observed. In conjunction to this, there can be seasonal effects, as well as specific holidays or events impacting traffic patterns.

**Value of inventory to our advertisers**

Certain types of advertiser verticals see very different response patterns at different times of day (noone orders food in the middle of the night), which influences when it makes sense to spend budget.

**Market price of inventory**
Finally, besides the availability of inventory and its value to the advertisers, the market price of the inventory will similarly impact when it makes sense to spend budget.

### Pacing
Based on the formulated spend plan, and keeping track of how we are tracking compared to the plan, we can dynamically update the plan going forward to catch up if we have underspent, or slow down if we have overspent.

For pacing then, we typically would use a PID controller (Start with I-term, consider whether the others help) that is fast to react and adjust the minROAS required.

If performance guarantees are important, it might be a one-sided controller that only can slow down spend (by enforcing higher expected ROAS), alternatively it can be a two-sided controller that also can adjust bidding upwards to ensure the budget is spent according to plan.

## Additional Challenges

### Explore-Exploit Tradeoff

### Feedback Loops
* Winner's curse / loser's curse
* Bid amount as instrumental variable to de-bias predictions

### Handling Margin
* Reduce bids compared to impressionvalue by the margin we want to keep
* If over-achieving performance targets; dynamically increase margins to boost profits and incentivise further spend

### Experimentation and Evaluation
* SUTVA violation
  * Market place experimentation
* Higher order impact
  * Change have first order impact, the different data generation process will then affect models as they retrain, until a new equilibrium is reached
* Switchback experimentation
* Intra-day split vs. next-day eval
* Aggregate impact across customers
* Spend vs. performance