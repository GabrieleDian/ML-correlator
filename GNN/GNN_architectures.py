"""
Advanced GNN architectures for planar graph classification
Designed for graphs with 11-16 nodes """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv, GATConv, GCNConv, GraphConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    SAGPooling
)
from torch_geometric.utils import  softmax



class GINNet(nn.Module):
    """
    Graph Isomorphism Network - More powerful than GCN for graph-level tasks
    GIN is provably as powerful as the WL test for graph isomorphism
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.2, num_layers=3):
        super(GINNet, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Jumping knowledge - combine features from all layers
        self.jump = nn.Linear(num_layers * hidden_dim, hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None, return_embedding: bool = False):
        # Store representations from each layer
        layer_representations = []

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_representations.append(x)

        # Jumping knowledge connection (node-level)
        x = torch.cat(layer_representations, dim=-1)
        x = self.jump(x)

        # If embedding requested, return node-level embedding pre-pooling
        if return_embedding:
            return x

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_add_pool(x, batch)
        return self.classifier(x)

    def node_embeddings(self, x, edge_index, batch=None):
        """Convenience wrapper: return node-level embeddings after self.jump (pre-pooling)."""
        return self.forward(x, edge_index, batch=batch, return_embedding=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    global_add_pool, global_mean_pool, global_max_pool
)
from torch_geometric.utils import softmax


############ GAT#################

class GATNet(nn.Module):
    """
    Performance-first GAT for graph classification, inspired by your GIN structure:
    - arbitrary num_layers (ModuleList)
    - Jumping Knowledge on nodes: concat all layer outputs -> projection
    - optional residual connections
    - flexible pooling: add/mean/max/att
    - attention dropout inside GAT + feature dropout outside
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
        num_layers: int = 3,
        num_heads: int = 4,
        attn_dropout: float | None = None,
        pooling: str = "att",          # "att" often strong on small graphs
        residual: bool = True,
        activation: str = "elu",       # "elu" is standard for GAT
        negative_slope: float = 0.2,
        jk_mode: str = "cat",          # "cat" (your GIN style) or "last"
    ):
        super().__init__()

        if attn_dropout is None:
            attn_dropout = dropout

        pooling = pooling.lower().strip()
        if pooling not in {"add", "mean", "max", "att"}:
            raise ValueError(f"pooling must be one of add/mean/max/att, got {pooling!r}")
        self.pooling = pooling

        activation = activation.lower().strip()
        if activation not in {"elu", "relu", "gelu"}:
            raise ValueError(f"activation must be one of elu/relu/gelu, got {activation!r}")
        self.activation = activation

        jk_mode = jk_mode.lower().strip()
        if jk_mode not in {"cat", "last"}:
            raise ValueError(f"jk_mode must be 'cat' or 'last', got {jk_mode!r}")
        self.jk_mode = jk_mode

        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.residual = bool(residual)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # for residual shape matching when needed

        # Strategy:
        # - Use concat=True for the first layer to expand capacity (hidden_dim * heads),
        #   then compress back to hidden_dim via a linear projection before next layer.
        # - From layer 2 onward, keep concat=False to maintain stable hidden_dim.
        #
        # This is often stronger than concat=False everywhere.
        self.pre_projs = nn.ModuleList()  # project expanded head output -> hidden_dim

        for i in range(self.num_layers):
            if i == 0:
                in_ch = num_features
                conv = GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,              # expand capacity early
                    dropout=attn_dropout,
                    negative_slope=negative_slope,
                    add_self_loops=True,
                    bias=True,
                )
                self.convs.append(conv)
                self.pre_projs.append(nn.Linear(hidden_dim * num_heads, hidden_dim))
            else:
                in_ch = hidden_dim
                conv = GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,             # keep dim stable
                    dropout=attn_dropout,
                    negative_slope=negative_slope,
                    add_self_loops=True,
                    bias=True,
                )
                self.convs.append(conv)
                self.pre_projs.append(nn.Identity())

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # residual projection (only needed if shapes differ; here they don't after pre_proj)
            self.res_projs.append(nn.Identity())

        # JK projection (node-level)
        if self.jk_mode == "cat":
            self.jump = nn.Linear(self.num_layers * hidden_dim, hidden_dim)
        else:
            self.jump = nn.Identity()

        # Global attention pooling head if needed
        self.global_att = nn.Linear(hidden_dim, 1) if self.pooling == "att" else None

        # Classifier (keep your proven style)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return F.elu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        return F.relu(x)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            return global_add_pool(x, batch)
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "max":
            return global_max_pool(x, batch)

        # "att"
        att = self.global_att(x)                 # (N,1)
        att = softmax(att, batch, dim=0)         # normalize per graph
        return global_add_pool(x * att, batch)

    def forward(self, x, edge_index, batch=None, return_embedding: bool = False):
        layer_representations = []
        h = x

        for i in range(self.num_layers):
            h_in = h

            # feature dropout (external)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # conv
            h = self.convs[i](h, edge_index)

            # if first layer expanded heads, compress to hidden_dim
            h = self.pre_projs[i](h)

            # BN + activation
            h = self.batch_norms[i](h)
            h = self._act(h)

            # residual (after normalization/act usually works well)
            if self.residual and h_in.shape[-1] == h.shape[-1]:
                h = h + h_in

            layer_representations.append(h)

        # Jumping knowledge (node-level)
        if self.jk_mode == "cat":
            h = torch.cat(layer_representations, dim=-1)
            h = self.jump(h)
        else:
            h = layer_representations[-1]

        if return_embedding:
            return h

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        hg = self._pool(h, batch)
        return self.classifier(hg)

    def node_embeddings(self, x, edge_index, batch=None):
        return self.forward(x, edge_index, batch=batch, return_embedding=True)


class HybridGNN(nn.Module):
    """
    Hybrid architecture combining multiple convolution types
    Captures different aspects of graph structure
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.2):
        super(HybridGNN, self).__init__()
        
        self.dropout = dropout
        
        # Different types of graph convolutions
        self.gcn = GCNConv(num_features, hidden_dim)
        self.gin = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.gat = GATConv(num_features, hidden_dim // 2, heads=2)
        
        # Combine different representations
        self.combine = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Second layer
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        
        # Pooling layers
        self.pool1 = SAGPooling(hidden_dim, ratio=0.8)
        
        # Final layers
        self.global_pool = nn.Linear(hidden_dim * 3, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply different convolutions
        x_gcn = F.relu(self.gcn(x, edge_index))
        x_gin = F.relu(self.gin(x, edge_index))
        x_gat = F.relu(self.gat(x, edge_index))
        
        # Concatenate representations
        x = torch.cat([x_gcn, x_gin, x_gat], dim=-1)
        x = self.combine(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Hierarchical pooling (optional)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
        # Multiple global pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Combine pooling results
        x = torch.cat([x_mean, x_max, x_add], dim=-1)
        x = self.global_pool(x)
        
        return self.classifier(x)

#####New Hybrid #############



# needed for HYbridGNNv2
class IdentityPooling(nn.Module):
    """No-op pooling with same return signature as SAGPooling."""
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return x, edge_index, edge_attr, batch, None, None

class HybridGNNv2(nn.Module):
    """
    Performance-first hybrid GNN:
    - Parallel "stem" block: GCN + GIN + GAT (learnable mixing)
    - Then a stack of message-passing layers (GraphConv by default)
    - BN + activation + dropout, residuals
    - Optional SAGPooling at configurable stages
    - Jumping Knowledge over stages (node-level)
    - Flexible pooling (add/mean/max/att or combo)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,

        # depth / capacity
        num_layers: int = 3,                # number of "stages" after stem
        stem_heads: int = 4,                # for GAT in stem
        stem_gat_out: int | None = None,    # per-head out channels; auto if None

        # behavior knobs
        residual: bool = True,
        activation: str = "relu",           # "relu" / "elu" / "gelu"
        jk_mode: str = "cat",               # "cat" or "last"
        pooling: str = "combo",             # "add"|"mean"|"max"|"att"|"combo"
        use_sag: bool = True,
        sag_ratio: float = 0.8,
        sag_every: int = 2,                 # apply SAGPooling every N stages (>=1)

        # attention pooling
        att_pool_hidden: int | None = None, # if pooling="att" or "combo"
    ):
        super().__init__()

        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.residual = bool(residual)

        activation = activation.lower().strip()
        if activation not in {"relu", "elu", "gelu"}:
            raise ValueError(f"activation must be one of relu/elu/gelu, got {activation!r}")
        self.activation = activation

        jk_mode = jk_mode.lower().strip()
        if jk_mode not in {"cat", "last"}:
            raise ValueError(f"jk_mode must be 'cat' or 'last', got {jk_mode!r}")
        self.jk_mode = jk_mode

        pooling = pooling.lower().strip()
        if pooling not in {"add", "mean", "max", "att", "combo"}:
            raise ValueError(f"pooling must be add/mean/max/att/combo, got {pooling!r}")
        self.pooling = pooling

        self.use_sag = bool(use_sag)
        self.sag_ratio = float(sag_ratio)
        self.sag_every = int(sag_every)

        # ---------- STEM: parallel convs on raw features ----------
        # GCN stem
        self.stem_gcn = GCNConv(num_features, hidden_dim)

        # GIN stem
        self.stem_gin = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )

        # GAT stem (use concat=True then project to hidden_dim)
        if stem_gat_out is None:
            # choose per-head out so total ~hidden_dim, but keep >=8
            stem_gat_out = max(8, hidden_dim // max(1, stem_heads))

        self.stem_gat = GATConv(
            in_channels=num_features,
            out_channels=stem_gat_out,
            heads=stem_heads,
            concat=True,
            dropout=dropout,
        )
        self.stem_gat_proj = nn.Linear(stem_gat_out * stem_heads, hidden_dim)

        # Learnable mixing of the 3 stem representations (stronger than concat+linear)
        # x = sum_i gate_i(x) * proj_i(x_i)
        self.stem_proj = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.stem_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.stem_bn = nn.BatchNorm1d(hidden_dim)

        # ---------- STAGES: deeper message passing ----------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.pools = nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            if self.use_sag and (self.sag_every > 0) and ((i + 1) % self.sag_every == 0):
                self.pools.append(SAGPooling(hidden_dim, ratio=self.sag_ratio))
            else:
                self.pools.append(IdentityPooling())


        # Jumping knowledge projection
        if self.jk_mode == "cat":
            self.jump = nn.Linear(self.num_layers * hidden_dim, hidden_dim)
        else:
            self.jump = nn.Identity()

        # Global attention pooling head (optional)
        if self.pooling in {"att", "combo"}:
            if att_pool_hidden is None:
                att_pool_hidden = hidden_dim
            self.global_att = nn.Sequential(
                nn.Linear(hidden_dim, att_pool_hidden),
                nn.ReLU(),
                nn.Linear(att_pool_hidden, 1),
            )
        else:
            self.global_att = None

        # Pooling combiner if needed
        if self.pooling == "combo":
            self.pool_combine = nn.Linear(hidden_dim * 4, hidden_dim)  # mean/max/add/att
        elif self.pooling in {"add", "mean", "max", "att"}:
            self.pool_combine = nn.Identity()
        else:
            raise RuntimeError("unreachable")

        # Classifier (keep your proven style)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "elu":
            return F.elu(x)
        return F.gelu(x)

    def _att_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # attention weights per-node per-graph
        att = self.global_att(x)              # (N,1)
        att = softmax(att, batch, dim=0)
        return global_add_pool(x * att, batch)

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            return global_add_pool(x, batch)
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "max":
            return global_max_pool(x, batch)
        if self.pooling == "att":
            return self._att_pool(x, batch)

        # combo: mean, max, add, att
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_att = self._att_pool(x, batch)
        return self.pool_combine(torch.cat([x_mean, x_max, x_add, x_att], dim=-1))

    def forward(self, x, edge_index, batch=None, return_embedding: bool = False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ----- STEM -----
        x_gcn = self._act(self.stem_gcn(x, edge_index))
        x_gin = self._act(self.stem_gin(x, edge_index))
        x_gat = self._act(self.stem_gat(x, edge_index))
        x_gat = self.stem_gat_proj(x_gat)

        # learnable mixing
        x_cat = torch.cat([x_gcn, x_gin, x_gat], dim=-1)
        gates = self.stem_gate(x_cat)                   # (N,3)
        gates = F.softmax(gates, dim=-1)                # convex weights

        x = gates[:, 0:1] * x_gcn + gates[:, 1:2] * x_gin + gates[:, 2:3] * x_gat
        x = self.stem_bn(x)
        x = self._act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- STAGES -----
        reps = []
        for i in range(self.num_layers):
            x_in = x

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = self._act(x)

            if self.residual:
                x = x + x_in

            # optional SAG pooling (coarsens graph)
            pool = self.pools[i]
            if pool is not None:
                x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)


            reps.append(x)

        # JK node embedding
        if self.jk_mode == "cat":
            x = torch.cat(reps, dim=-1)
            x = self.jump(x)
        else:
            x = reps[-1]

        if return_embedding:
            return x

        # Global pool to graph embedding
        hg = self._global_pool(x, batch)
        return self.classifier(hg)

    def node_embeddings(self, x, edge_index, batch=None):
        return self.forward(x, edge_index, batch=batch, return_embedding=True)


class PlanarGNN(nn.Module):
    """
    Specialized architecture for planar graphs
    Incorporates domain-specific inductive biases
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.2):
        super(PlanarGNN, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Planar-aware convolutions (2 layers optimal for small graphs)
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        )
        
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        )
        
        # Edge features processing (for planar graphs)
        self.edge_attr_lin = nn.Linear(1, hidden_dim)
        
        # Skip connections
        self.skip_lin = nn.Linear(num_features + hidden_dim * 2, hidden_dim)
        
        # Readout with attention
        self.att_weight = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.att_weight)
        
        # Final classifier with residual
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Store input for skip connection
        x_input = x
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # First convolution
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second convolution
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        
        # Skip connection combining input and both conv outputs
        x_skip = torch.cat([x_input, x1, x2], dim=-1)
        x_skip = self.skip_lin(x_skip)
        
        # Attention-based readout
        att_scores = torch.matmul(x_skip, self.att_weight.t())
        att_scores = softmax(att_scores, batch, dim=0)
        
        # Weighted and unweighted pooling
        x_att = global_add_pool(x_skip * att_scores, batch)
        x_mean = global_mean_pool(x_skip, batch)
        
        # Combine both pooling strategies
        x = torch.cat([x_att, x_mean], dim=-1)
        
        return self.classifier(x)


class SimpleButEffectiveGNN(nn.Module):
    """
    Simple architecture that often works best for small graphs
    Based on the principle that simpler is better for limited data
    """
    def __init__(self, num_features, hidden_dim=32, num_classes=2, dropout=0.1):
        super(SimpleButEffectiveGNN, self).__init__()
        
        # Just two GIN layers with batch norm
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Two layers only (optimal for small graphs)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # Combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        return self.classifier(x)


def create_gnn_model(
    architecture: str = 'gin',
    num_features: int = 26,
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.2,
    **kwargs
):
    """
    Factory function to create different GNN architectures.

    Args:
        architecture: One of ['gin', 'gat', 'hybrid', 'hybrid_v2', 'planar', 'simple']
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        dropout: Dropout rate
        **kwargs: Additional architecture-specific parameters
    """

    architectures = {
        'gin': GINNet,

        # IMPORTANT:
        # - Point 'gat' to the NEW improved GATNet (arbitrary layers + JK + optional pooling/residual)
        'gat': GATNet,

        # Keep old hybrid as 'hybrid' OR swap to v2 (your choice)
        # Option A (recommended): keep both
        'hybrid': HybridGNN,          # old
        'hybrid_v2': HybridGNNv2,     # improved

        'planar': PlanarGNN,
        'simple': SimpleButEffectiveGNN,
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(architectures.keys())}")

    model_class = architectures[architecture]

    # Allowed kwargs per architecture (for sweeps, prevents silent typos)
    valid_params_map = {
        'gin': {
            'num_layers',
        },
        'gat': {
            # new GATNet knobs (keep the ones you actually want to sweep)
            'num_layers',
            'num_heads',
            'attn_dropout',
            'pooling',          # "add" or "att" or "combo" depending on your GATNet version
            'residual',
            'activation',
            'jk_mode',
            'negative_slope',
        },
        'hybrid': {
            # old HybridGNN has no extra params currently
        },
        'hybrid_v2': {
            # HybridGNNv2 knobs
            'num_layers',
            'stem_heads',
            'stem_gat_out',
            'residual',
            'activation',
            'jk_mode',
            'pooling',          # "add"|"mean"|"max"|"att"|"combo"
            'use_sag',
            'sag_ratio',
            'sag_every',
            'att_pool_hidden',
        },
        'planar': set(),
        'simple': set(),
    }

    valid_params = valid_params_map.get(architecture, set())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_class(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        **filtered_kwargs
    )

# Example usage and comparison
if __name__ == "__main__":
    # Example parameters
    num_features = 26  # From your feature extractor
    batch_size = 32

    # Create different models (performance-oriented configs)
    models = {
        # Strong baseline (your current champion)
        "GIN": create_gnn_model(
            "gin",
            num_features=num_features,
            hidden_dim=64,
            dropout=0.2,
            num_layers=3,
        ),

        # Improved GAT (new version: arbitrary layers + JK + residual + pooling knobs)
        # Try both add pooling (fairer vs GIN) and attention pooling (often stronger).
        "GAT (add pool, JK, residual)": create_gnn_model(
            "gat",
            num_features=num_features,
            hidden_dim=64,
            dropout=0.2,
            num_layers=3,
            num_heads=4,
            pooling="add",
            residual=True,
            jk_mode="cat",
            activation="elu",
        ),
        "GAT (att pool, JK, residual)": create_gnn_model(
            "gat",
            num_features=num_features,
            hidden_dim=64,
            dropout=0.2,
            num_layers=3,
            num_heads=4,
            pooling="att",
            residual=True,
            jk_mode="cat",
            activation="elu",
        ),

        # Hybrid improved (keep old 'hybrid' too if you want, but v2 is the performance candidate)
        "Hybrid v2 (combo pool, no SAG)": create_gnn_model(
            "hybrid_v2",
            num_features=num_features,
            hidden_dim=64,
            dropout=0.2,
            num_layers=3,
            stem_heads=4,
            pooling="combo",
            use_sag=False,
            residual=True,
            jk_mode="cat",
            activation="relu",
        ),
        "Hybrid v2 (combo pool, with SAG)": create_gnn_model(
            "hybrid_v2",
            num_features=num_features,
            hidden_dim=64,
            dropout=0.2,
            num_layers=3,
            stem_heads=4,
            pooling="combo",
            use_sag=True,
            sag_every=2,
            sag_ratio=0.8,
            residual=True,
            jk_mode="cat",
            activation="relu",
        ),

        # Existing baselines
        "Planar-Specific": create_gnn_model(
            "planar",
            num_features=num_features,
            hidden_dim=64,
            dropout=0.2,
        ),
        "Simple but Effective": create_gnn_model(
            "simple",
            num_features=num_features,
            hidden_dim=32,
            dropout=0.2,
        ),
    }

    # Print model summaries
    print("Model Architectures Summary:")
    print("=" * 72)
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<34} {num_params:>12,} parameters")

    print("\nRecommended comparison protocol:")
    print("1) Same split, same seed, same training budget (epochs/early-stop), same scheduler.")
    print("2) Run 3-10 seeds per model (these small graphs can be high-variance).")
    print("3) Report mean±std ROC-AUC + best checkpoint ROC-AUC; keep calibration metrics too.")
    print("4) For GAT/Hybrid, sweep a small grid: num_layers {2,3,4}, heads {2,4,8}, dropout {0.0-0.3}.")
    print("5) Keep pooling as a sweep dimension (add vs att vs combo), not a fixed choice.")
