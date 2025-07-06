import warnings
import networkx as nx
import pandas as pd
import numpy as np
from scipy.linalg import eigvalsh

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def laplacian_eigenvalues(G, k=10):
    try:
        L = nx.normalized_laplacian_matrix(G).toarray()
        return list(np.sort(eigvalsh(L))[:k]) + [np.nan] * (k - G.number_of_nodes())
    except Exception:
        return [np.nan] * k


def adjacency_eigenvalues(G, k=5):
    try:
        A = nx.adjacency_matrix(G).toarray()
        eigs = np.sort(eigvalsh(A))[::-1]
        eigs = list(eigs[:k]) + [np.nan] * (k - len(eigs))
        gap = eigs[0] - eigs[1] if not np.isnan(eigs[1]) else np.nan
        return eigs, gap
    except Exception:
        return [np.nan] * k, np.nan


# ------------------------------------------------------------------
# per-graph feature extractor
# ------------------------------------------------------------------
def compute_graph_features(edges, name="graph"):
    G = nx.Graph()
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))      # ← removes self-loops

    # early-exit template for empty graph
    if G.number_of_nodes() == 0:
        base = {f"{name}_{col}": np.nan for col in [
            "n_nodes","n_edges","density",
            "avg_degree","max_degree","min_degree","degree_var","degree_entropy",
            "deg_hist_0","deg_hist_1","deg_hist_2","deg_hist_3","deg_hist_4","deg_hist_5","deg_hist_6",
            "avg_clustering","transitivity","n_triangles","n_squares",
            "core_max","core_mean","core_entropy",
            "dist_q10","dist_q50","dist_q90","diameter","avg_shortest_path",
            "assortativity","edge_betw_mean","edge_betw_std",
            "betw_mean","betw_std","close_mean","close_std",
            "adj_spec_gap","wl_hash"
        ]}
        base.update({f"{name}_lap_eigval_{i+1}": np.nan for i in range(10)})
        base.update({f"{name}_adj_eig_{i+1}": np.nan for i in range(5)})
        return pd.Series(base)

    feats = {}

    # -- basic counts
    feats[f"{name}_n_nodes"]  = G.number_of_nodes()
    feats[f"{name}_n_edges"]  = G.number_of_edges()
    feats[f"{name}_density"]  = nx.density(G)

    # -- degree stats
    degs = np.array([d for _, d in G.degree()])
    feats[f"{name}_avg_degree"] = degs.mean()
    feats[f"{name}_max_degree"] = degs.max()
    feats[f"{name}_min_degree"] = degs.min()
    feats[f"{name}_degree_var"] = degs.var()

    counts = np.bincount(degs)
    p = counts[counts > 0] / counts.sum()
    feats[f"{name}_degree_entropy"] = -(p * np.log2(p)).sum()

    hist, _ = np.histogram(degs, bins=[0,1,2,3,4,5,6,np.inf])
    for i, h in enumerate(hist):
        feats[f"{name}_deg_hist_{i}"] = int(h)

    # -- clustering & motifs
    feats[f"{name}_avg_clustering"] = nx.average_clustering(G)
    feats[f"{name}_transitivity"]   = nx.transitivity(G)
    feats[f"{name}_n_triangles"]    = sum(nx.triangles(G).values()) // 3
    feats[f"{name}_n_squares"]      = len([c for c in nx.cycle_basis(G) if len(c) == 4])

    # -- k-core
    core_vals = np.array(list(nx.core_number(G).values()))
    feats[f"{name}_core_max"]      = core_vals.max()
    feats[f"{name}_core_mean"]     = core_vals.mean()
    pc = np.bincount(core_vals); pc = pc / pc.sum()
    feats[f"{name}_core_entropy"]  = -(pc * np.log2(pc + 1e-12)).sum()

    # -- path length quantiles
    if nx.is_connected(G):
        dists = [l for src in nx.all_pairs_shortest_path_length(G) for l in src[1].values()]
        feats[f"{name}_dist_q10"]  = np.quantile(dists, 0.10)
        feats[f"{name}_dist_q50"]  = np.quantile(dists, 0.50)
        feats[f"{name}_dist_q90"]  = np.quantile(dists, 0.90)
        feats[f"{name}_diameter"]  = nx.diameter(G)
        feats[f"{name}_avg_shortest_path"] = np.mean(dists)
    else:
        feats.update({f"{name}_dist_q10": np.nan, f"{name}_dist_q50": np.nan,
                      f"{name}_dist_q90": np.nan, f"{name}_diameter": np.nan,
                      f"{name}_avg_shortest_path": np.nan})

    # -- assortativity
    try:
        feats[f"{name}_assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        feats[f"{name}_assortativity"] = np.nan

    # -- edge betweenness
    eb = list(nx.edge_betweenness_centrality(G).values())
    feats[f"{name}_edge_betw_mean"] = np.mean(eb)
    feats[f"{name}_edge_betw_std"]  = np.std(eb)

    # -- node centralities
    bc_vals = list(nx.betweenness_centrality(G).values())
    cc_vals = list(nx.closeness_centrality(G).values())
    feats[f"{name}_betw_mean"]  = np.mean(bc_vals)
    feats[f"{name}_betw_std"]   = np.std(bc_vals)
    feats[f"{name}_close_mean"] = np.mean(cc_vals)
    feats[f"{name}_close_std"]  = np.std(cc_vals)

    # -- spectra
    for i, val in enumerate(laplacian_eigenvalues(G, k=10)):
        feats[f"{name}_lap_eigval_{i+1}"] = val
    adj_eigs, gap = adjacency_eigenvalues(G, k=5)
    for i, val in enumerate(adj_eigs):
        feats[f"{name}_adj_eig_{i+1}"] = val
    feats[f"{name}_adj_spec_gap"] = gap

    # -- WL hash
    feats[f"{name}_wl_hash"] = nx.weisfeiler_lehman_graph_hash(G)

    return pd.Series(feats)


# ------------------------------------------------------------------
# two-graph combiner
# ------------------------------------------------------------------
def compute_combined_graph_features(edges1, edges2):
    g1 = compute_graph_features(edges1, name="G1")
    g2 = compute_graph_features(edges2, name="G2")

    diff, ratio = {}, {}
    eps = 1e-6
    for k in g1.index:
        if not k.startswith("G1_"):        # only numeric G1 keys
            continue
        k2 = k.replace("G1_", "G2_")
        if k2 not in g2:
            continue
        v1, v2 = g1[k], g2[k2]
        if isinstance(v1, (int, float, np.floating)) and isinstance(v2, (int, float, np.floating)):
            if not (pd.isna(v1) or pd.isna(v2)):
                diff[f"diff_{k[3:]}"]  = v1 - v2
                ratio[f"ratio_{k[3:]}"] = v1 / (v2 + eps)
            else:
                diff[f"diff_{k[3:]}"]  = np.nan
                ratio[f"ratio_{k[3:]}"] = np.nan

    return pd.concat([g1, g2, pd.Series(diff), pd.Series(ratio)]).to_frame().T



from pathlib import Path
from tqdm import tqdm
if __name__ == "__main__":

    for j in tqdm(range(5,10)):
        df = pd.read_csv(Path(f"../ML-correlator/Graph_Edge_Data/graph_data_{j}.csv"))

        res = []
        for i in tqdm(range(0, len(df))):
            g0 = eval(df['DEN_EDGES'].iloc[i])
            g1 = eval(df['NUM_EDGES'].iloc[i])
            features_df = compute_combined_graph_features(g0, g1)
            res.append(features_df)

        res = pd.concat(res).reset_index()
        res.to_csv(f'/Users/rezadoobary/Documents/ML-correlator/Tree classifier for graphs/mixed_loops/features_{j}.csv')

