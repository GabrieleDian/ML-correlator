import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class GraphBuilder:
    """
    Enhanced GraphBuilder for planar graphs.
    Supports feature selection and separates node-level from graph-level features.
    """
    
    # Feature groups for easy selection
    FEATURE_GROUPS = {
        'basic': ['degree'],
        'face': ['num_faces', 'avg_face_size', 'max_face_size', 'face_size_variance'],
        'spectral_node': ['fiedler_vector', 'eigenvector_energy', 'third_eigenvector'],
        'spectral_global': ['algebraic_connectivity', 'spectral_gap', 'largest_eigenvalue'],
        'dual': ['dual_degree', 'dual_clustering', 'dual_degree_ratio', 'dual_betweenness', 'face_edge_ratio'],
        'centrality': ['betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 
                      'clustering_coefficient', 'pagerank'],
        'laplacian_pe': ['laplacian_eigenvectors']  
    }
    
    def __init__(self, solid_edges, coeff, node_labels=None, 
                 selected_features: Optional[List[str]] = None,
                 include_global_features: bool = False,
                 laplacian_pe_k: int = 4):
        """
        Args:
            solid_edges: List of edge tuples
            coeff: Target coefficient for the graph
            node_labels: Optional node labels, auto-inferred if None
            selected_features: List of feature groups to include. If None, uses default set.
            include_global_features: Whether to include graph-level features as node features
            laplacian_pe_k: Number of Laplacian eigenvectors for positional encoding
        """
        # Auto-infer node labels if not provided
        if node_labels is None:
            node_labels = sorted(set(u for e in solid_edges for u in e))
        
        self.node_labels = node_labels
        self.label2idx = {label: i for i, label in enumerate(node_labels)}
        self.solid_edges = solid_edges
        self.num_nodes = len(self.node_labels)
        self.y = torch.tensor(coeff, dtype=torch.long)
        
        # Feature selection
        if selected_features is None:
            # Default feature set optimized for small planar graphs
            self.selected_features = ['basic', 'face', 'spectral_node', 'centrality', 'laplacian_pe']
        else:
            self.selected_features = selected_features
            
        self.include_global_features = include_global_features
        
        # Initialize eigenvector transform
        self.laplacian_pe_k = laplacian_pe_k
        self.eigen_transform = AddLaplacianEigenvectorPE(k=laplacian_pe_k, attr_name=None)
        
        # Track feature dimensions for each group
        self.feature_dims = {
            'basic': 1,
            'face': 4,
            'spectral_node': 3,
            'spectral_global': 3,
            'dual': 5,
            'centrality': 5,  # Removed degree_centrality to avoid redundancy
            'laplacian_pe': laplacian_pe_k
        }
    
    def _get_networkx_graph(self, edge_index):
        """Convert to NetworkX for analysis"""
        edge_list = edge_index.t().numpy().tolist()
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
        """Extract face-based features for planar graphs"""
        face_features = np.zeros((self.num_nodes, 4))
        
        # Get planar embedding
        is_planar, embedding = nx.check_planarity(G)
        if not is_planar:
            raise ValueError("Graph is not planar!")
        
        # Extract faces from the planar embedding
        faces = []
        visited_edges = set()
        
        for node in embedding.nodes():
            for neighbor in embedding.neighbors(node):
                edge = (node, neighbor) if node < neighbor else (neighbor, node)
                if edge not in visited_edges:
                    face = list(embedding.traverse_face(node, neighbor))
                    faces.append(face)
                    # Mark edges of this face as visited
                    for i in range(len(face)):
                        u, v = face[i], face[(i + 1) % len(face)]
                        visited_edges.add((u, v) if u < v else (v, u))
        
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
    
    def _extract_spectral_features(self, G):
        """Extract spectral features, separating node-level and graph-level"""
        node_features = np.zeros((self.num_nodes, 3))
        global_features = np.zeros(3)
        
        # Get Laplacian matrix
        L = nx.laplacian_matrix(G).todense()
        eigenvals, eigenvecs = np.linalg.eigh(L)
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Global spectral properties
        global_features[0] = eigenvals[1] if len(eigenvals) > 1 else 0  # Algebraic connectivity
        global_features[1] = eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0  # Spectral gap
        global_features[2] = eigenvals[-1]  # Largest eigenvalue
        
        # Node-level spectral features
        for i in range(self.num_nodes):
            # Fiedler vector value (important for graph partitioning)
            if len(eigenvals) > 1:
                node_features[i, 0] = eigenvecs[i, 1]
            
            # Sum of squares of eigenvector components (spectral energy)
            node_features[i, 1] = np.sum(eigenvecs[i, :min(5, len(eigenvals))]**2)
            
            # Third eigenvector component
            if len(eigenvals) > 2:
                node_features[i, 2] = eigenvecs[i, 2]
        
        return node_features, global_features
    
    def _extract_dual_graph_features(self, G):
        """Extract features based on dual graph properties"""
        dual_features = np.zeros((self.num_nodes, 5))
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            
            # Count triangles (faces in original become edges in dual)
            triangles = 0
            for i, u in enumerate(neighbors):
                for v in neighbors[i+1:]:
                    if G.has_edge(u, v):
                        triangles += 1
            
            # Dual degree approximation
            dual_degree = max(1, triangles + len(neighbors) - 2)
            
            # Dual clustering approximation
            if len(neighbors) > 1:
                dual_clustering = triangles / (len(neighbors) * (len(neighbors) - 1) / 2)
            else:
                dual_clustering = 0
            
            # Store features
            dual_features[node, 0] = dual_degree
            dual_features[node, 1] = dual_clustering
            
            # Ratio of dual degree to original degree
            orig_degree = G.degree(node)
            if orig_degree > 0:
                dual_features[node, 2] = dual_degree / orig_degree
            
            # Dual betweenness approximation
            dual_features[node, 3] = dual_degree * dual_clustering
            
            # Face-edge ratio
            if orig_degree > 0:
                dual_features[node, 4] = dual_degree / (orig_degree + 1)
        
        return dual_features
    
    def _extract_centrality_features(self, G):
        """Extract various centrality measures (excluding degree centrality)"""
        centrality_features = np.zeros((self.num_nodes, 5))
        
        # Compute centralities
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        clustering = nx.clustering(G)
        pagerank = nx.pagerank(G)
        
        # Eigenvector centrality with fixed iterations
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06)
        except nx.PowerIterationFailedConvergence:
            # Use degree centrality as fallback
            eigenvector_cent = nx.degree_centrality(G)
        
        for i in range(self.num_nodes):
            centrality_features[i, 0] = betweenness_cent[i]
            centrality_features[i, 1] = closeness_cent[i]
            centrality_features[i, 2] = eigenvector_cent[i]
            centrality_features[i, 3] = clustering[i]
            centrality_features[i, 4] = pagerank[i]
        
        return centrality_features
    
    def build(self, extra_node_features=None):
        """
        Build enhanced graph with selected features.
        
        Args:
            extra_node_features: Additional node features to include
            
        Returns:
            Data object with selected features
        """
        # Create edge index
        edge_list = []
        for u, v in self.solid_edges:
            i, j = self.label2idx[u], self.label2idx[v]
            edge_list += [[i, j], [j, i]]  # bidirectional
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create NetworkX graph for analysis
        G = self._get_networkx_graph(edge_index)
        
        # Collect selected features
        all_features = []
        feature_names = []
        global_features_dict = {}
        
        # Extract features based on selection
        if 'basic' in self.selected_features:
            degree_feat = degree(edge_index[0], num_nodes=self.num_nodes).view(-1, 1)
            all_features.append(degree_feat.numpy())
            feature_names.append('degree')
        
        if 'face' in self.selected_features:
            face_features = self._extract_face_features(G)
            all_features.append(face_features)
            feature_names.extend(['num_faces', 'avg_face_size', 'max_face_size', 'face_size_variance'])
        
        if 'spectral_node' in self.selected_features or 'spectral_global' in self.selected_features:
            node_spectral, global_spectral = self._extract_spectral_features(G)
            
            if 'spectral_node' in self.selected_features:
                all_features.append(node_spectral)
                feature_names.extend(['fiedler_vector', 'eigenvector_energy', 'third_eigenvector'])
            
            if 'spectral_global' in self.selected_features:
                global_features_dict['spectral'] = global_spectral
                if self.include_global_features:
                    # Broadcast global features to all nodes
                    global_broadcast = np.tile(global_spectral, (self.num_nodes, 1))
                    all_features.append(global_broadcast)
                    feature_names.extend(['algebraic_connectivity', 'spectral_gap', 'largest_eigenvalue'])
        
        if 'dual' in self.selected_features:
            dual_features = self._extract_dual_graph_features(G)
            all_features.append(dual_features)
            feature_names.extend(['dual_degree', 'dual_clustering', 'dual_degree_ratio', 
                                'dual_betweenness', 'face_edge_ratio'])
        
        if 'centrality' in self.selected_features:
            centrality_features = self._extract_centrality_features(G)
            all_features.append(centrality_features)
            feature_names.extend(['betweenness_centrality', 'closeness_centrality', 
                                'eigenvector_centrality', 'clustering_coefficient', 'pagerank'])
        
        # Add extra features if provided
        if extra_node_features is not None:
            assert extra_node_features.shape[0] == self.num_nodes, \
                "extra_node_features must match number of nodes"
            all_features.append(extra_node_features.numpy() if torch.is_tensor(extra_node_features) 
                              else extra_node_features)
            feature_names.extend([f'extra_{i}' for i in range(extra_node_features.shape[1])])
        
        # Concatenate all features
        if all_features:
            x = torch.FloatTensor(np.concatenate(all_features, axis=1))
        else:
            # If no features selected, use degree as default
            x = degree(edge_index[0], num_nodes=self.num_nodes).view(-1, 1).float()
            feature_names = ['degree']
        
        # Create data object
        data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes, y=self.y)
        
        # Store feature names and global features as attributes
        data.feature_names = feature_names
        data.global_features = global_features_dict
        
        # Apply Laplacian eigenvector transform if selected
        if 'laplacian_pe' in self.selected_features:
            data = self.eigen_transform(data)
            # Update feature names to include Laplacian PE
            data.feature_names.extend([f'laplacian_pe_{i}' for i in range(self.laplacian_pe_k)])
        
        return data
    
    def get_feature_info(self):
        """Get information about the features that will be extracted"""
        info = {
            'selected_features': self.selected_features,
            'total_dimensions': sum(self.feature_dims[group] for group in self.selected_features 
                                  if group in self.feature_dims),
            'feature_groups': {group: self.FEATURE_GROUPS[group] for group in self.selected_features 
                             if group in self.FEATURE_GROUPS}
        }
        return info

def create_graph_dataset(graphs_data: List[Tuple[List, int]], 
                        feature_config) -> List[Data]:
    """
    Create dataset from graph data using improved GraphBuilder
    
    Args:
        graphs_data: List of (edges, label) tuples
        feature_config: Configuration for GraphBuilder
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    
    dataset = []
    all_features = []
    
    # First pass: collect all features for normalization
    print("Extracting features...")
    for edges, label in graphs_data:
        builder = GraphBuilder(
            solid_edges=edges,
            coeff=label,
            **feature_config
        )
        data = builder.build()
        dataset.append(data)
        all_features.append(data.x.numpy())
    
    # Compute normalization statistics
    all_features = np.vstack(all_features)
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Second pass: normalize features
    print("Normalizing features...")
    for i, data in enumerate(dataset):
        # Get the starting index for this graph's features
        start_idx = sum(d.num_nodes for d in dataset[:i])
        end_idx = start_idx + data.num_nodes
        
        # Normalize
        normalized_features = scaler.transform(data.x.numpy())
        data.x = torch.FloatTensor(normalized_features)
    
    print(f"Created dataset with {len(dataset)} graphs")
    print(f"Feature dimensions: {dataset[0].x.shape[1]}")
    print(f"Feature names: {dataset[0].feature_names}")
    
    return dataset, scaler





# Example usage:
if __name__ == "__main__":
    # Example graph with 12 nodes
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'),
             ('A', 'E'), ('B', 'F'), ('C', 'G'), ('D', 'H'),
             ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'E'),
             ('E', 'I'), ('F', 'J'), ('G', 'K'), ('H', 'L'),
             ('I', 'J'), ('J', 'K'), ('K', 'L'), ('L', 'I')]
    
    # Create builder with custom feature selection
    builder = GraphBuilder(
        solid_edges=edges,
        coeff=1,
        selected_features=['basic', 'face', 'spectral_node', 'centrality'],
        include_global_features=False
    )
    
    # Get feature information
    print("Feature Info:", builder.get_feature_info())
    
    # Build the graph
    data = builder.build()
    print(f"Graph with {data.num_nodes} nodes and {data.x.shape[1]} features")
    print(f"Feature names: {data.feature_names}")
    print(f"Global features: {data.global_features}")