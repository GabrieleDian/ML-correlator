# Musings

- Mixed loop 0/1 coefficients on denominator graphs only - going from 5 loops to 10 loops

- Derive cusp equations that can be used as features:
    
    - Physics-aware features engineering: for every l-loop graph g in trainingset - compute cusp fingerprint:
    
        - will current graph g appear in higher loop order. e.g. a 5 loop row could be : [1,graph_features] to indicate the this g is a parent of a bunch of graphs in (l+1) that the graph reduces to.

        - can also add automorphism weights.

        - So this has already been done here.
        - https://arxiv.org/pdf/2410.09859
        - https://arxiv.org/pdf/2503.15593v1

        - Idea set up the linear system of equations and start to solve. Might be good to identify what we are unable to find by solving these.

- Apply as a soft-constraint loss function
    
    - Something like $\lambda | Ac_{\text{pred}}^{(l+1)} - b_{\text{true}}^{(l)}|$. where A codifies the relationship between the parent coefficient and the children.
    
    - problem: Is this useful if I don't already have the l+1 data? 

    - Has been used here: https://arxiv.org/pdf/2305.17387v2


- Broader question:
    - Information flow. There is a clearly a way to fix the coefficients (at least being 0,1) for l given l-1. These are physics informed. 

    - How are the remaining coefficients fixed? Is it a topological/geometric property, or more physics?

    - How can we test this?


- Key consideration:
    - Large amounts of coefficient data is already wiped out due to triangle rule. The idea here is to predict the coefficeints as a graph quantity.

    - For a proposal at 13 loops do we want to make us of the cusp rule?