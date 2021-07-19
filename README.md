# **Stochastic optimization models for a bike-sharing problem with transshipment**

The problem in exam refers to a bike-sharing provider who needs to manage a fleet of bikes over a set of bike-stations, each with given capacity and stochastic demand. here the focus is on One-way bike sharing with transshipment in which:

1. The user can pick up a bike at a station and drop it off at a different station
2. Transshipment of bikes among stations is performed at the end of the day, to have the optimal number of bikes at each station at the beginning of the service on the next day.

For this problem a two-stage stochastic optimization model is proposed _(heuristic to be added)_

## Problem Description

We consider the problem faced by a bike-sharing service provider who needs to **manage** a **ﬂeet of bikes** over a set of bike-stations with given capacities, in order to serve the **stochastic rental demand** over space and time. A **unit procurement cost** is paid for each bike assigned to each station at the beginning of the service. The operational time frame is one day. The **delivery** of bikes to bike-stations is **assumed** to be **instantaneous** (lead time equal to zero), as this operation can be carried out before the start of the service. Backlogging is not allowed. A **unit stock-out cost** is paid if realized demand exceeds the number of bikes assigned to a station, and a **unit transshipment cost** is paid at the end of the rentals, when the bike-station inventory levels are rebalanced. Each bike rental demand is deﬁned by an origin-destination pair, where the destination is unknown to the provider at the time of rental. A *stochastic demand to each origin-destination pair* is assigned. The rent must start at the user-deﬁned time period or it is lost, determining a shortage for the provider and a reduction of the service level for the user. A shortage realizes when a rental demand arises in a bike-station, but no bikes are available: The user quits the service and looks towards an alternative transportation mode. A shortage causes a cost increase, a reduced service level and a reduced likelihood of future rental requests. The number of bikes that cannot be left in a bike-station as it is full when the user arrives at the station determines an overﬂow: The user cannot quit the service until the bike is redirected and positioned by the user in the nearest bike-station with available capacity. An over-ﬂow causes a waste of time for the user and a cost. Our ***aim*** is to determine the ***number of bikes to assign to each bike-station at the beginning of the service***, in order to **minimize the expected total costs**, given by the sum of the procurement costs, the expected stock-out costs for unmet demand, the expected time-waste costs for overﬂow and the expected transshipment costs for repositioning bikes at the end of the service.


## Formulation
Define following notation


### Sets:
* $\mathcal{B}$ : set of bike-stations, $\mathcal{B} = \{1, \dots , B\}$

* $\mathcal{S}$ : set of scenarios, $\mathcal{S} = \{1, \dots , S\}$ or finite set of possible realizations of uncertainty


### Deterministic Parameters:
* $c \in \mathbb{R}^{+} $ : procurement cost per bike at each bike-station at the beginning of the service;
* $v_i \in \mathbb{R}^{+} $ : stock-out cost per bike at bike-station, $i \in \mathcal{B} $;
* $w_i \in \mathbb{R}^{+} $ : time-waste cost per bike due to overﬂow at bike-station, $i \in \mathcal{B} $;
* $t_{ij} \in \mathbb{R}^{+} $: unit transshipment cost per bike transshipped from bike-station $i$ to bike-station $j$, with $i, j \in \mathcal{B} $;
* $k_i \in \mathbb{Z}^{+} $ : capacity of bike-station $i \in \mathcal{B} $;

### Stochastic Parameters

Let $(\Xi, \mathcal{A}, p)$ be a probability space with $\Xi$ set of outcomes, $\sigma$-algebra $\mathcal{A}$, probability $p$ and $\xi \in \Xi$ a particular outcome representing the rental demand on each origin destination pair of bike-station. We define:

* $\xi_{ijs} \in \Xi \subset \mathbb{Z}^{+}$: rental demand from bike-station $i$ to bike-station $j$ in scenario, $s$, with $i, j \in \mathcal{B}, s \in \mathcal{S}$
* $p_s \in [0, 1 ]$: probability of scenario $s \in \mathcal{S}$. Notice that $\sum^{S}_{s=1} p_s = 1$

#### First stage variables:

* $x_i$: number of bikes to assign to bike-station  $i \in \mathcal{B}$ at the beginning of the service. 


Here you solve the deterministic problem in order to find the optimal number of bikes to assign to each bike-station. 

We denote it with $x_i^*$ the **optimal solution**. From it we take only the first-stage decision vector over all bike-stations **x**$ = [x_1, \dots, x_B]^T$

#### Second Stage Variables

After the placement of the bikes, the stochastic demands $\xi_{ijs} $ occurr on each origin-destination pair $i, j \in \mathcal{B}$ and the minimum of the available and request bikes is actually rented (per objective function)

The decision variables are:

* $\beta_{ijs}$: Number of rented bikes from bike-station $i$ to bike-station $j$ in scenario $s$;
* $I^{+}_{is}$:realized surplus of bikes at bike-station $i$ in scenario $s$. Surplus does not cost anything to the provider;
* $I^{-}_{ijs}$: realized shortage of bikes at origin-destination pair $i, j$ in scenario $s$;
* $\rho_{ijs}$: Number of redirected bikes from bike-station $i$ to bike-station $j$ in scenario $s$;
* $O^{+}_{is}$: Redisual capacity at bike-station $i$ in scenario $s$;
* $O^{-}_{is}$: Overflow at bike-station $i$ n scenario $s$;
* $\tau_{ijs}$: Number of transshipped bikes from bike-station $i$ to bike-station $j$ in scenario $s$; 
* $T^{+}_{is}$: Excess of bikes at bike-station $i$ in scenario $s$;
* $T^{-}_{is}$: Lack of bikes at bike-station $i$ in scenario $s$

## Project

the problem can be formulated as the following two-stage integer stochastic program:
$$
\min c \sum_{i=1}^{\mathcal{B}} x_i + \sum_{s=1}^{\mathcal{S}} p_s \sum_{i=1}^{\mathcal{B}} \big[v_i \sum_{j=1}^{\mathcal{B}} I^{-}_{ijs} + w_i O^{-}_{is} + \sum_{j=1}^{\mathcal{B}} t_{ij} \tau_{ijs} \big]
$$
subject to:
$$
\begin{array}{lcl}
x_i \leq k_i & \quad & \forall i \in \mathcal{B}\\
\\
\beta_{ijs} = \xi_{ijs}-I^{-}_{ijs} & \quad & \forall i,j \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
I^{+}_{is} - \sum^{B}_{j=1} I^{-}_{ijs} = x_i - \sum^{B}_{j=1} \xi_{ijs}& \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
O^{+}_{is} - O^{-}_{is} = k_i - x_i + \sum^{B}_{j=1} \beta_{ijs} - \sum^{B}_{j=1} \beta_{jis} & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
\sum^{B}_{j=1} \rho_{ijs} = O^{-}_{is} & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
\sum^{B}_{j=1} \rho_{jis} \leq O^{+}_{is} & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
T^{+}_{is} - T^{-}_{is} = k_i - O^{+}_{is} + \sum^{B}_{j=1} \rho_{jis} - x_i & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S}
\\
\sum^{B}_{j=1} \tau_{jis} = T^{+}_{is} & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
\sum^{B}_{j=1} \tau_{jis} = T^{-}_{is} & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S} \\
\\
\end{array}
$$

Existence Domain:
$$
\begin{array}{lcl}
x_i, I^{+}_{is}, O^{+}_{is},O^{-}_{is},T^{+}_{is},T^{-}_{is}, \in \mathbb{Z}^{+} & \quad & \forall i \in \mathcal{B}, \forall s \in \mathcal{S}
\\
\tau_{ijs}, \beta_{ijs},\rho_{ijs},I^{-}_{ijs}, \in \mathbb{Z}^{+} & \quad & \forall i, j \in \mathcal{B}, \forall s \in \mathcal{S}
\\
\end{array}
$$
