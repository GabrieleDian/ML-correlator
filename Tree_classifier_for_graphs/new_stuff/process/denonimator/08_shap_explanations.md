# Motif3 Feature Definitions

This document provides precise mathematical descriptions and intuitive interpretations of the **Motif3 feature family** used in the graph learning pipeline.

Motif3 features quantify **triadic structure**, capturing both:

- **Closed triplets (Triangles)**
- **Open triplets (Wedges)**

Together, these metrics characterize clustering organization, closure dynamics, and structural heterogeneity.

---

## Conceptual Overview

Triadic structures are fundamental building blocks of graph topology:

- **Triangles → Structural closure**
- **Wedges → Potential closure**

Motif3 features describe:

✔ Local clustering intensity  
✔ Closure vs branching balance  
✔ Edge participation patterns  
✔ Structural heterogeneity  

---

# Feature Definitions (Technically Precise)

---

### Multi-Triangle Edge Fraction  
**Original Name:** `Motif_triangle_edge_frac_ge2`

**Definition:**  
Fraction of edges participating in **at least two distinct triangles**.

**Formal Interpretation:**  
Measures redundant local clustering and multi-triangle connectivity.

**Structural Meaning:**  
Identifies edges embedded within densely clustered regions.

---

### Triangle-Free Edge Fraction  
**Original Name:** `Motif_triangle_edge_frac_zero`

**Definition:**  
Fraction of edges not participating in **any triangle**.

**Formal Interpretation:**  
Captures prevalence of tree-like or non-clustered connectivity.

**Structural Meaning:**  
Indicates absence of local closure around edges.

---

### Mean Triangle Incidence per Edge  
**Original Name:** `Motif_triangle_edge_incidence_mean`

**Definition:**  
Mean number of triangles incident per edge.

**Formal Interpretation:**  
Average edge-level clustering intensity.

**Structural Meaning:**  
Typical level of closure surrounding edges.

---

### Median Triangle Incidence per Edge  
**Original Name:** `Motif_triangle_edge_incidence_median`

**Definition:**  
Median number of triangles incident per edge.

**Formal Interpretation:**  
Robust central tendency of edge clustering participation.

**Structural Meaning:**  
Typical clustering excluding extreme edges.

---

### Upper-Tail Triangle Incidence (Q90)  
**Original Name:** `Motif_triangle_edge_incidence_q90`

**Definition:**  
90th percentile of triangle incidence per edge.

**Formal Interpretation:**  
Characterizes high-clustering edge regime.

**Structural Meaning:**  
Detects highly clustered structural cores.

---

### Triangle Incidence Variability  
**Original Name:** `Motif_triangle_edge_incidence_std`

**Definition:**  
Standard deviation of triangle incidence per edge.

**Formal Interpretation:**  
Quantifies heterogeneity of local clustering structure.

**Structural Meaning:**  
Indicates uniform vs localized clustering.

---

### Total Triangle Count  
**Original Name:** `Motif_triangles`

**Definition:**  
Total number of triangles in the graph.

**Formal Interpretation:**  
Absolute measure of three-node closure density.

**Structural Meaning:**  
Raw clustering magnitude.

---

### Normalized Triangle Density  
**Original Name:** `Motif_triangles_per_Cn3`

**Definition:**  
Triangle count normalized by the number of possible node triples.

**Formal Interpretation:**  
Scale-invariant global clustering density.

**Structural Meaning:**  
Clustering independent of graph size.

---

### Total Wedge Count  
**Original Name:** `Motif_wedges`

**Definition:**  
Total number of connected node triplets (open triads).

**Formal Interpretation:**  
Measures potential closure opportunities.

**Structural Meaning:**  
Latent triangle formation capacity.

---

### Normalized Wedge Density  
**Original Name:** `Motif_wedges_per_max`

**Definition:**  
Wedge count normalized by its theoretical maximum.

**Formal Interpretation:**  
Scale-adjusted measure of open triadic structure.

**Structural Meaning:**  
Relative branching vs closure organization.

---

# Intuitive Interpretations

Motif3 features describe how graphs organize connectivity at the triadic level:

- **Triangles → Closed structures**
- **Wedges → Near-triangles**

Key interpretations:

✔ Clustering intensity  
✔ Redundancy of local structure  
✔ Presence of cohesive cores  
✔ Structural heterogeneity  
✔ Closure vs branching balance  

---

# Scientific Role in the Model

Motif3 features capture **local structural organization**, complementing global descriptors such as spectral features.

They are particularly informative for:

✔ Mesoscale structure  
✔ Community cohesion  
✔ Local redundancy  
✔ Network densification dynamics  
✔ Scaling effects in larger graphs  

---

# Recommended Citation Language

For manuscripts and reports:

> **“Motif3 features characterize the balance between open and closed triadic structures, capturing clustering intensity, closure dynamics, and heterogeneity of local connectivity patterns.”**

Or:

> **“Triangle-based metrics quantify structural closure, while wedge statistics capture latent closure opportunities and branching organization.”**

---

# Summary

Motif3 features provide a multi-faceted description of triadic structure:

✔ Closure magnitude  
✔ Closure density  
✔ Edge participation patterns  
✔ Structural heterogeneity  
✔ Branching vs clustering balance  

These metrics are critical for understanding **local organization and scaling behavior** in graph-based learning systems.



-------------


# Centrality Feature Definitions

This document provides formal definitions and interpretations of the **Centrality feature family** used in the graph learning pipeline.

Centrality features summarize the distributional properties of node-level centrality measures across the graph. Rather than describing individual nodes, these variables encode **graph-level structural signatures derived from node importance profiles**.

Each centrality metric captures a distinct aspect of network organization:

✔ Path mediation (Betweenness)  
✔ Global accessibility (Closeness)  
✔ Recursive influence (Eigenvector Centrality)

For each metric, multiple statistical descriptors are computed to characterize the **shape, variability, and extremal structure** of the node distribution.

---

# Betweenness Centrality Features

Betweenness centrality quantifies how frequently a node lies on shortest paths between other node pairs.

**Structural Role:**  
Captures mediation, brokerage, and structural bottlenecks.

---

### **Centrality_betweenness_mean**

**Definition:**  
Mean betweenness centrality across all nodes.

**Interpretation:**  
Average level of path mediation in the graph.

**Structural Meaning:**  
Higher values indicate stronger reliance on shortest-path routing.

---

### **Centrality_betweenness_std**

**Definition:**  
Standard deviation of node betweenness values.

**Interpretation:**  
Heterogeneity of mediation roles.

**Structural Meaning:**  
High values imply uneven distribution of structural control.

---

### **Centrality_betweenness_skew**

**Definition:**  
Skewness of the betweenness distribution.

**Interpretation:**  
Asymmetry of mediation dominance.

**Structural Meaning:**  
Positive skew → Few nodes dominate shortest-path flow.

---

### **Centrality_betweenness_max**

**Definition:**  
Maximum betweenness centrality observed.

**Interpretation:**  
Strength of the most critical bottleneck node.

**Structural Meaning:**  
Identifies dominant structural bridges.

---

# Closeness Centrality Features

Closeness centrality measures inverse average shortest-path distance to all other nodes.

**Structural Role:**  
Captures global reachability and network integration.

---

### **Centrality_closeness_mean**

**Definition:**  
Mean closeness centrality across nodes.

**Interpretation:**  
Average accessibility of nodes.

**Structural Meaning:**  
Higher values → More compact / efficiently connected graphs.

---

### **Centrality_closeness_std**

**Definition:**  
Standard deviation of closeness values.

**Interpretation:**  
Variability of global accessibility.

**Structural Meaning:**  
High values → Presence of peripheral vs central nodes.

---

### **Centrality_closeness_skew**

**Definition:**  
Skewness of closeness distribution.

**Interpretation:**  
Asymmetry in node accessibility.

**Structural Meaning:**  
Positive skew → Few highly central nodes.

---

### **Centrality_closeness_max**

**Definition:**  
Maximum closeness centrality observed.

**Interpretation:**  
Accessibility of the most globally central node.

**Structural Meaning:**  
Detects dominant network hubs.

---

# Normalized Closeness Features

Normalized closeness rescales values to reduce graph-size dependence.

✔ Improves cross-graph comparability  
✔ Controls for trivial scaling effects

---

### **Centrality_closeness_mean_norm**

Mean normalized closeness.

**Meaning:**  
Graph compactness independent of node count.

---

### **Centrality_closeness_max_norm**

Maximum normalized closeness.

**Meaning:**  
Dominance of the most globally accessible node.

---

# Eigenvector Centrality Features

Eigenvector centrality measures recursive influence based on connections to important nodes.

**Structural Role:**  
Captures hierarchical influence propagation.

---

### **Centrality_eigenvector_mean**

**Definition:**  
Mean eigenvector centrality across nodes.

**Interpretation:**  
Average influence propagation capacity.

---

### **Centrality_eigenvector_std**

**Definition:**  
Standard deviation of eigenvector values.

**Interpretation:**  
Influence inequality / hierarchy strength.

---

### **Centrality_eigenvector_skew**

**Definition:**  
Skewness of eigenvector distribution.

**Interpretation:**  
Dominance of influential nodes.

**Structural Meaning:**  
Positive skew → Few nodes control influence structure.

---

### **Centrality_eigenvector_max**

**Definition:**  
Maximum eigenvector centrality observed.

**Interpretation:**  
Strength of dominant influence hub.

---

# Structural Interpretation Summary

| Metric | Captures |
|----------|-------------|
| Betweenness | Path mediation / bottlenecks |
| Closeness | Global accessibility / compactness |
| Eigenvector | Recursive influence / hierarchy |

Distributional statistics quantify:

✔ Structural inequality  
✔ Role concentration  
✔ Network heterogeneity  
✔ Hierarchical organization

---

# Scientific Role in the Model

Centrality features encode **mesoscale graph organization**, bridging:

✔ Local structure (motifs)  
✔ Global geometry (spectral features)

They are particularly sensitive to:

✔ Bottleneck formation  
✔ Hub dominance  
✔ Connectivity heterogeneity  
✔ Graph compactness

---

# Recommended Citation Language

> **“Centrality features summarize graph-level structural organization via distributional statistics of node betweenness, closeness, and eigenvector centrality.”**

Or:

> **“These descriptors quantify mediation structure, accessibility heterogeneity, and influence hierarchy.”**

---

# Summary

Centrality features provide graph-scale representations of node roles:

✔ Structural bottlenecks  
✔ Accessibility structure  
✔ Influence hierarchy  
✔ Role inequality  
✔ Network integration