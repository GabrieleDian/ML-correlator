import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
from torch_geometric.transforms import AddLaplacianEigenvectorPE

class GraphBuilder:
    def __init__(self, solid_edges, coeff, node_labels=None):
        # Auto-infer node labels if not provided
        if node_labels is None:
            node_labels = sorted(set(u for e in solid_edges for u in e))
        self.node_labels = node_labels
        self.label2idx = {label: i for i, label in enumerate(node_labels)}
        self.solid_edges = solid_edges
        self.num_nodes = len(self.node_labels)
        self.y = torch.tensor(coeff, dtype=torch.long)
        
        # Initialize eigenvector transform
        self.eigen_transform = AddLaplacianEigenvectorPE(k=4, attr_name=None)
    
    def _get_networkx_graph(self, edge_index):
        """Convert to NetworkX for analysis"""
        edge_list = edge_index.t().numpy().tolist()
        # Remove duplicate edges for undirected graph
        unique_edges = []
        seen = set()
        for u, v in edge_list:
            if (u, v) not in seen and (v, u) not in seen:
                unique_edges.append((u, v))
                seen.add((u, v))
        
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(unique_edges)
        return G
    
    def _extract_face_features(self, G):
        """Extract face-based features for planar graphs using cycle detection"""
        face_features = np.zeros((self.num_nodes, 4))  # 4 face-related features per node
        
        try:
            # Check if graph is planar
            if not nx.is_planar(G):
                # For non-planar graphs, use cycle-based approximation
                return self._extract_cycle_features(G)
            
            # Try to get planar embedding using NetworkX's planar layout
            try:
                # Get the planar embedding using the correct NetworkX method
                embedding, _ = nx.check_planarity(G)
                if embedding is None:
                    return self._extract_cycle_features(G)
                
                # Extract faces from the planar embedding
                faces = []
                visited_edges = set()
                
                for node in embedding.nodes():
                    for neighbor in embedding.neighbors(node):
                        edge = (node, neighbor) if node < neighbor else (neighbor, node)
                        if edge not in visited_edges:
                            try:
                                face = list(embedding.traverse_face(node, neighbor))
                                faces.append(face)
                                # Mark edges of this face as visited
                                for i in range(len(face)):
                                    u, v = face[i], face[(i + 1) % len(face)]
                                    visited_edges.add((u, v) if u < v else (v, u))
                            except Exception:
                                continue
                
                # Calculate face-based features for each node
                for i in range(self.num_nodes):
                    node_faces = [f for f in faces if i in f]
                    
                    # Feature 1: Number of faces containing this node
                    face_features[i, 0] = len(node_faces)
                    
                    # Feature 2: Average face size
                    if node_faces:
                        face_features[i, 1] = np.mean([len(f) for f in node_faces])
                    
                    # Feature 3: Maximum face size
                    if node_faces:
                        face_features[i, 2] = max([len(f) for f in node_faces])
                    
                    # Feature 4: Face size variance
                    if len(node_faces) > 1:
                        face_sizes = [len(f) for f in node_faces]
                        face_features[i, 3] = np.var(face_sizes)
                
                return face_features
                
            except Exception:
                # If planar embedding fails, fall back to cycle features
                return self._extract_cycle_features(G)
            
        except Exception:
            # If anything fails, use cycle-based approximation
            return self._extract_cycle_features(G)
    
    def _extract_cycle_features(self, G):
        """Fallback method using cycle-based features when planar embedding fails"""
        face_features = np.zeros((self.num_nodes, 4))
        
        try:
            # Use a simpler approach: find cycles using basis
            try:
                cycle_basis = nx.cycle_basis(G)
            except Exception:
                cycle_basis = []
            
            # Also try to find triangles and small cycles manually
            all_cycles = list(cycle_basis)
            
            # Add triangles explicitly
            triangles = [list(triangle) for triangle in nx.triangles(G) if nx.triangles(G)[triangle] > 0]
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                for i, u in enumerate(neighbors):
                    for v in neighbors[i+1:]:
                        if G.has_edge(u, v):
                            triangle = [node, u, v]
                            if triangle not in all_cycles:
                                all_cycles.append(triangle)
            
            # Calculate cycle-based features for each node
            for i in range(self.num_nodes):
                node_cycles = [c for c in all_cycles if i in c]
                
                # Feature 1: Number of cycles containing this node
                face_features[i, 0] = len(node_cycles)
                
                # Feature 2: Average cycle size
                if node_cycles:
                    face_features[i, 1] = np.mean([len(c) for c in node_cycles])
                else:
                    face_features[i, 1] = 0.0
                
                # Feature 3: Maximum cycle size
                if node_cycles:
                    face_features[i, 2] = max([len(c) for c in node_cycles])
                else:
                    face_features[i, 2] = 0.0
                
                # Feature 4: Cycle size variance
                if len(node_cycles) > 1:
                    cycle_sizes = [len(c) for c in node_cycles]
                    face_features[i, 3] = np.var(cycle_sizes)
                else:
                    face_features[i, 3] = 0.0
        
        except Exception:
            # If cycle detection also fails, use simple triangle counting
            try:
                clustering_coeffs = nx.clustering(G)
                triangles_dict = nx.triangles(G)
                
                for i in range(self.num_nodes):
                    if i in triangles_dict:
                        triangles = triangles_dict[i]
                        face_features[i, 0] = triangles
                        face_features[i, 1] = 3.0 if triangles > 0 else 0.0
                        face_features[i, 2] = 3.0 if triangles > 0 else 0.0
                        face_features[i, 3] = 0.0
                    else:
                        face_features[i, :] = 0.0
            except Exception:
                # Last resort: all zeros
                pass
        
        return face_features
    
    def _extract_advanced_spectral_features(self, G):
        """Extract advanced spectral features beyond basic eigenvectors"""
        spectral_features = np.zeros((self.num_nodes, 6))
        
        try:
            # Get Laplacian matrix
            L = nx.laplacian_matrix(G).todense()
            eigenvals, eigenvecs = np.linalg.eigh(L)
            
            # Sort eigenvalues and eigenvectors
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Global spectral properties (same for all nodes)
            algebraic_connectivity = eigenvals[1] if len(eigenvals) > 1 else 0
            spectral_gap = eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0
            largest_eigenval = eigenvals[-1]
            
            # Spectral centrality measures
            for i in range(self.num_nodes):
                # Feature 1: Algebraic connectivity (Fiedler value)
                spectral_features[i, 0] = algebraic_connectivity
                
                # Feature 2: Spectral gap
                spectral_features[i, 1] = spectral_gap
                
                # Feature 3: Largest eigenvalue
                spectral_features[i, 2] = largest_eigenval
                
                # Feature 4: Fiedler vector value (important for graph partitioning)
                if len(eigenvals) > 1:
                    spectral_features[i, 3] = eigenvecs[i, 1]  # Fiedler vector
                
                # Feature 5: Sum of squares of eigenvector components
                spectral_features[i, 4] = np.sum(eigenvecs[i, :min(5, len(eigenvals))]**2)
                
                # Feature 6: Spectral clustering coefficient
                if len(eigenvals) > 2:
                    spectral_features[i, 5] = eigenvecs[i, 2]  # Third eigenvector
        
        except Exception:
            # If spectral analysis fails, use default values
            pass
        
        return spectral_features
    
    def _extract_dual_graph_features(self, G):
        """Extract features from the dual graph representation"""
        dual_features = np.zeros((self.num_nodes, 5))
        
        try:
            # For planar graphs, we can construct the dual graph
            # In the dual graph, faces become nodes and adjacency represents shared edges
            
            # Create a dual graph based on triangles and local structure
            dual_degrees = {}
            dual_clustering = {}
            
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                
                # Count triangles (faces in original become edges in dual)
                triangles = 0
                for i, u in enumerate(neighbors):
                    for v in neighbors[i+1:]:
                        if G.has_edge(u, v):
                            triangles += 1
                
                # Dual degree approximation
                dual_degrees[node] = max(1, triangles + len(neighbors) - 2)
                
                # Dual clustering approximation
                if len(neighbors) > 1:
                    dual_clustering[node] = triangles / (len(neighbors) * (len(neighbors) - 1) / 2)
                else:
                    dual_clustering[node] = 0
            
            # Extract dual graph features
            for i in range(self.num_nodes):
                # Feature 1: Dual degree
                dual_features[i, 0] = dual_degrees.get(i, 0)
                
                # Feature 2: Dual clustering coefficient
                dual_features[i, 1] = dual_clustering.get(i, 0)
                
                # Feature 3: Ratio of dual degree to original degree
                orig_degree = G.degree(i)
                if orig_degree > 0:
                    dual_features[i, 2] = dual_degrees.get(i, 0) / orig_degree
                
                # Feature 4: Dual betweenness approximation
                dual_features[i, 3] = dual_degrees.get(i, 0) * dual_clustering.get(i, 0)
                
                # Feature 5: Face-edge ratio
                if orig_degree > 0:
                    dual_features[i, 4] = dual_degrees.get(i, 0) / (orig_degree + 1)
        
        except Exception:
            # If dual graph analysis fails, use default values
            pass
        
        return dual_features
    
    def _extract_centrality_features(self, G):
        """Extract various centrality measures"""
        centrality_features = np.zeros((self.num_nodes, 6))
        
        try:
            # Degree centrality
            degree_cent = nx.degree_centrality(G)
            
            # Betweenness centrality
            betweenness_cent = nx.betweenness_centrality(G)
            
            # Closeness centrality
            closeness_cent = nx.closeness_centrality(G)
            
            # Eigenvector centrality
            try:
                eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                eigenvector_cent = {i: 0 for i in G.nodes()}
            
            # Clustering coefficient
            clustering = nx.clustering(G)
            
            # PageRank
            pagerank = nx.pagerank(G)
            
            for i in range(self.num_nodes):
                centrality_features[i, 0] = degree_cent.get(i, 0)
                centrality_features[i, 1] = betweenness_cent.get(i, 0)
                centrality_features[i, 2] = closeness_cent.get(i, 0)
                centrality_features[i, 3] = eigenvector_cent.get(i, 0)
                centrality_features[i, 4] = clustering.get(i, 0)
                centrality_features[i, 5] = pagerank.get(i, 0)
        
        except Exception:
            # If centrality analysis fails, use default values
            pass
        
        return centrality_features
    
    def build(self, extra_node_features=None):
        """Build enhanced graph with planar-specific features"""
        # Create edge index
        edge_list = []
        for u, v in self.solid_edges:
            i, j = self.label2idx[u], self.label2idx[v]
            edge_list += [[i, j], [j, i]]  # bidirectional
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Basic degree feature
        degree_feat = degree(edge_index[0], num_nodes=self.num_nodes).view(-1, 1)
        
        # Create NetworkX graph for advanced analysis
        G = self._get_networkx_graph(edge_index)
        
        # Extract all advanced features with error handling
        face_features = self._extract_face_features(G)
        spectral_features = self._extract_advanced_spectral_features(G)
        dual_features = self._extract_dual_graph_features(G)
        centrality_features = self._extract_centrality_features(G)
        
        # Combine all features
        all_features = [
            degree_feat.numpy(),
            face_features,
            spectral_features,
            dual_features,
            centrality_features
        ]
        
        # Add extra features if provided
        if extra_node_features is not None:
            assert extra_node_features.shape[0] == self.num_nodes, \
                "extra_node_features must match number of nodes"
            all_features.append(extra_node_features.numpy() if torch.is_tensor(extra_node_features) 
                              else extra_node_features)
        
        # Concatenate all features
        x = torch.FloatTensor(np.concatenate(all_features, axis=1))
        
        # Create initial data object
        data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes, y=self.y)
        
        # Apply Laplacian eigenvector transform (this adds the top 4 eigenvectors)
        try:
            data = self.eigen_transform(data)
        except Exception:
            # If eigenvector transform fails, continue without it
            pass
        
        return data
