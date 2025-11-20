import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from .MeekSTV import MeekCore
import string
import textwrap
from .utils import convert_pf_to_numpy_arrays

class MeekGraph:
    def __init__(self, profile, m, auditable_margins_per_deg, use_numerical_labels=False, label_edges_by_idx=None):
        self.profile = profile
        self.ballot_matrix, self.mult_vec, self.fpv_vec = convert_pf_to_numpy_arrays(profile)
        self.m = m
        self.num_cands = len(profile.candidates)
        self.auditable_margins_per_deg = auditable_margins_per_deg
        self.candidates = list(profile.candidates)
        if label_edges_by_idx is None:
            self.use_numerical_labels = use_numerical_labels
        else:
            self.use_numerical_labels = label_edges_by_idx
        
        # Initialize NetworkX graph
        self.tree = nx.DiGraph()
        
        # Keep track of nodes by layer and their coordinates
        self.nodes_by_layer = {}
        self.node_positions = {}
        self.node_counter = 0
        
        # Node ID mapping: node_id -> node_data
        self.node_lookup = {}
        
        # Build the tree
        self.build_tree()
    
    def node_to_string(self, node):
        """Convert a node dict to a unique string identifier"""
        winners = sorted(node["winner_to_cand"])
        losers = sorted(node["initial_losers"])
        return f"W:{','.join(map(str, winners))}_L:{','.join(map(str, losers))}"
    
    def get_node_id(self, layer, position_in_layer):
        """Generate node ID in format NX (e.g., 0A, 1B, 2C, etc.)"""
        if position_in_layer < 26:
            letter = string.ascii_uppercase[position_in_layer]
        else:
            # Handle more than 26 nodes per layer (unlikely but safe)
            letter = f"Z{position_in_layer - 25}"
        return f"{layer}{letter}"
    
    def string_to_readable(self, node_str):
        """Convert node string to human-readable format using candidate names"""
        if node_str == "W:_L:":
            return "Root"
        
        parts = node_str.split('_')
        winners_part = parts[0][2:]  # Remove 'W:'
        losers_part = parts[1][2:]   # Remove 'L:'
        
        winner_names = [self.candidates[int(i)] for i in winners_part.split(',') if i]
        loser_names = [self.candidates[int(i)] for i in losers_part.split(',') if i]
        
        result = ""
        if winner_names:
            result += f"W:{','.join(winner_names)}"
        if loser_names:
            if result:
                result += " "
            result += f"L:{','.join(loser_names)}"
        
        return result if result else "Root"
    
    def add_node_to_graph(self, node, layer):
        """Add a node to the graph with proper positioning"""
        node_str = self.node_to_string(node)
        
        if node_str not in self.tree:
            # Add to layer tracking
            if layer not in self.nodes_by_layer:
                self.nodes_by_layer[layer] = []
            
            # Calculate position
            position_in_layer = len(self.nodes_by_layer.get(layer, []))
            node_id = self.get_node_id(layer, position_in_layer)
            x_position = position_in_layer
            y_position = -layer  # Negative so root is at top
            
            self.nodes_by_layer[layer].append(node_str)
            self.node_positions[node_str] = (x_position, y_position)
            
            # Store node lookup information
            self.node_lookup[node_id] = {
                'node_str': node_str,
                'winner_to_cand': node["winner_to_cand"].copy(),
                'initial_losers': node["initial_losers"].copy(),
                'layer': layer,
                'position_in_layer': position_in_layer
            }
            
            # Add to NetworkX graph with attributes
            self.tree.add_node(node_str, 
                             layer=layer,
                             winners=node["winner_to_cand"].copy(),
                             losers=node["initial_losers"].copy(),
                             readable=self.string_to_readable(node_str),
                             node_id=node_id,
                             degree=len(node["winner_to_cand"]))
        
        return node_str
    
    def _compute_layer_positions(self, horizontal_span, vertical_spacing):
        """Position nodes by layer with even spacing and centered alignment."""
        positions = {}
        if not self.nodes_by_layer:
            return positions
        
        effective_span = max(horizontal_span, 1.0)
        for layer in sorted(self.nodes_by_layer.keys()):
            nodes = self.nodes_by_layer[layer]
            layer_width = len(nodes)
            if layer_width == 0:
                continue
            if layer_width == 1:
                x_positions = [0.0]
            else:
                total_span = effective_span
                step = total_span / (layer_width - 1)
                start = -total_span / 2.0
                x_positions = [start + i * step for i in range(layer_width)]
            y = -layer * vertical_spacing
            for node_str, x in zip(nodes, x_positions):
                positions[node_str] = (x, y)
        
        # Guarantee every node has a coordinate (safety net for unexpected data).
        for node in self.tree.nodes:
            positions.setdefault(node, (0.0, 0.0))
        
        return positions
    
    def _draw_directional_edge_labels(self, positions, font_size, vertical_spacing):
        """Place edge labels along each edge with direction-aware offsets."""
        ax = plt.gca()
        _ = vertical_spacing  # preserved for API compatibility / future tweaks
        xs = [pos[0] for pos in positions.values()] or [0.0]
        ys = [pos[1] for pos in positions.values()] or [0.0]
        h_span = max(max(xs) - min(xs), 1e-6)
        v_span = max(max(ys) - min(ys), 1e-6)
        for u, v, data in self.tree.edges(data=True):
            label = data.get('label')
            if not label:
                continue
            x1, y1 = positions.get(u, (0.0, 0.0))
            x2, y2 = positions.get(v, (0.0, 0.0))
            dx = x2 - x1
            dy = y2 - y1
            ndx = dx / h_span
            ndy = dy / v_span
            length = (ndx ** 2 + ndy ** 2) ** 0.5
            if length < 1e-9:
                frac = 0.5
            else:
                horizontal_component = ndx / length  # -1 (left) to 1 (right)
                frac = 0.5 + 0.2 * horizontal_component
                frac = max(0.1, min(0.9, frac))
            text_x = x1 + frac * dx
            text_y = y1 + frac * dy
            ax.text(text_x,
                    text_y,
                    label,
                    fontsize=font_size,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=0.2))

    def _format_edge_label(self, label, max_width=14):
        """Insert line breaks into long labels to keep them readable."""
        if not label:
            return label
        if "\n" in label:
            return label
        if len(label) <= max_width:
            return label
        return textwrap.fill(label, width=max_width)

    def get_edge_label(self, from_node, to_node, use_numerical=False):
        """Generate edge label describing the transition"""
        from_winners = set(from_node["winner_to_cand"])
        to_winners = set(to_node["winner_to_cand"])
        from_losers = set(from_node["initial_losers"])
        to_losers = set(to_node["initial_losers"])
        
        new_winners = to_winners - from_winners
        new_losers = to_losers - from_losers
        
        label = ""
        if new_winners:
            candidate_idx = list(new_winners)[0]
            if use_numerical:
                label = f"Elect {candidate_idx}"
            else:
                candidate_name = self.candidates[candidate_idx]
                label = f"Elect {candidate_name}"
        elif new_losers:
            candidate_idx = list(new_losers)[0]
            if use_numerical:
                label = f"Eliminate {candidate_idx}"
            else:
                candidate_name = self.candidates[candidate_idx]
                label = f"Eliminate {candidate_name}"
        return self._format_edge_label(label)

    def next_possible_nodes(self, node):
        """Find next possible nodes and add them to the graph"""
        # Check if we already have enough winners - stop if we do
        if len(node["winner_to_cand"]) >= self.m:
            return []  # Stop expanding this branch
            
        try:
            current_core = MeekCore(
                ballot_matrix=self.ballot_matrix,
                mult_vec=self.mult_vec,
                m=self.m,
                num_cands=self.num_cands,
                winner_to_cand=node["winner_to_cand"],
                initial_losers=node["initial_losers"],
            )
            tallies, helper_vecs, play, tiebreak = current_core._run_first_round()
        except Exception as e:
            print(f"Error processing node {self.string_to_readable(self.node_to_string(node))}: {e}")
            return []
            
        deg = len(node["winner_to_cand"])
        auditable_margin = self.auditable_margins_per_deg[deg]
        quota = play["quota"]

        current_node_str = self.node_to_string(node)
        current_layer = len(node["winner_to_cand"]) + len(node["initial_losers"])
        possible_nodes = []

        tallies_with_masked_winners = tallies.copy()
        tallies_with_masked_winners[node["winner_to_cand"]] = -1

        # Check for mandatory winners (tallies significantly above quota)
        if np.any(tallies_with_masked_winners > quota + auditable_margin):
            only_possible_winner = np.argmax(tallies_with_masked_winners)
            only_possible_node = {
                "winner_to_cand": node["winner_to_cand"] + [only_possible_winner],
                "initial_losers": node["initial_losers"],
            }
            possible_nodes = [only_possible_node]
        else:
            # Check for possible winners (tallies close to quota)
            if np.any(tallies_with_masked_winners > quota - auditable_margin):
                possible_winner_mask = tallies_with_masked_winners > quota - auditable_margin
                biggest_tally = np.argmax(tallies_with_masked_winners)
                within_margin_of_biggest_mask = np.where(
                    possible_winner_mask & (tallies_with_masked_winners >= tallies_with_masked_winners[biggest_tally] - auditable_margin)
                )[0]
                
                for possible_winner in within_margin_of_biggest_mask:
                    possible_node = {
                        "winner_to_cand": node["winner_to_cand"] + [possible_winner],
                        "initial_losers": node["initial_losers"],
                    }
                    possible_nodes.append(possible_node)
            
            # Check for possible losers (tallies close to minimum)
            # Only consider candidates not already winners or losers
            active_candidates = set(range(self.num_cands)) - set(node["winner_to_cand"]) - set(node["initial_losers"])
            if active_candidates:
                active_tallies = tallies[list(active_candidates)]
                if len(active_tallies) > 0:
                    lowest_tally = np.min(active_tallies)
                    within_margin_of_lowest_mask = np.where(tallies <= lowest_tally + auditable_margin)[0]
                    
                    for possible_loser in within_margin_of_lowest_mask:
                        if possible_loser in active_candidates:
                            possible_node = {
                                "winner_to_cand": node["winner_to_cand"],
                                "initial_losers": node["initial_losers"] + [possible_loser],
                            }
                            possible_nodes.append(possible_node)

        # Add nodes to graph and create edges with labels
        for possible_node in possible_nodes:
            next_node_str = self.add_node_to_graph(possible_node, current_layer + 1)
            edge_label = self.get_edge_label(node, possible_node, use_numerical=self.use_numerical_labels)
            self.tree.add_edge(current_node_str, next_node_str, label=edge_label)

        return possible_nodes

    def build_tree(self):
        """Build the complete tree layer by layer"""
        # Start with root node
        root_node = {"winner_to_cand": [], "initial_losers": []}
        root_node_str = self.add_node_to_graph(root_node, 0)
        
        # Build tree layer by layer
        current_layer_nodes = [root_node]
        
        # Continue until we've explored all possibilities or reached max winners
        max_iterations = self.num_cands  # Safety limit
        for layer in range(max_iterations):
            next_layer_nodes = []
            
            for node in current_layer_nodes:
                possible_nodes = self.next_possible_nodes(node)
                next_layer_nodes.extend(possible_nodes)
            
            # Remove duplicates for next iteration
            seen = set()
            unique_next_nodes = []
            for node in next_layer_nodes:
                node_str = self.node_to_string(node)
                if node_str not in seen:
                    seen.add(node_str)
                    unique_next_nodes.append(node)
            
            current_layer_nodes = unique_next_nodes
            
            if not current_layer_nodes:
                break
    
    def lookup_node(self, node_id):
        """Look up a node by its ID (e.g., '0A', '1B', etc.)"""
        if node_id in self.node_lookup:
            info = self.node_lookup[node_id]
            winners = [self.candidates[i] for i in info['winner_to_cand']]
            losers = [self.candidates[i] for i in info['initial_losers']]
            
            return {
                'node_id': node_id,
                'layer': info['layer'],
                'position_in_layer': info['position_in_layer'],
                'winner_to_cand': info['winner_to_cand'],
                'initial_losers': info['initial_losers'],
                'winners': winners,
                'losers': losers
            }
        else:
            available_ids = sorted(self.node_lookup.keys())
            raise ValueError(f"Node ID '{node_id}' not found. Available IDs: {available_ids}")
    
    def list_nodes(self):
        """List all node IDs in the tree"""
        return sorted(self.node_lookup.keys())
    
    def plot_graph(self, figsize=(15, 10), node_size=3000, font_size=8, label_nodes=False, 
                   label_edges=True, title=None, spacing_multiplier=3.0, vertical_spacing=None):
        """Plot the tree with layers arranged vertically"""
        if not self.tree.nodes():
            print("No nodes to plot")
            return
            
        plt.figure(figsize=figsize)
        
        # Calculate spacing parameters based on layer widths
        max_layer_width = max((len(nodes) for nodes in self.nodes_by_layer.values()), default=1)
        if max_layer_width > 1:
            horizontal_span = spacing_multiplier * (max_layer_width - 1)
        else:
            horizontal_span = spacing_multiplier
        computed_vertical_spacing = spacing_multiplier * 1.5
        if vertical_spacing is None:
            vertical_spacing = computed_vertical_spacing
        
        # Compute layout so nodes are evenly spread in each layer
        adjusted_positions = self._compute_layer_positions(horizontal_span, vertical_spacing)
        if not adjusted_positions:
            print("No layout data found; using spring layout as fallback.")
            adjusted_positions = nx.spring_layout(self.tree, seed=42)
        
        # Create labels - either node IDs or readable names
        labels = None
        if label_nodes:
            labels = {node_str: self.tree.nodes[node_str]['node_id'] for node_str in self.tree.nodes()}
        
        # Determine node colors by degree (number of winners at that node)
        node_degrees = [self.tree.nodes[node].get('degree', 0) for node in self.tree.nodes()]
        max_degree = max(node_degrees, default=0)
        cmap = plt.cm.get_cmap('viridis')
        if max_degree == 0:
            node_colors = [cmap(0.3) for _ in node_degrees]
        else:
            node_colors = [cmap(0.2 + 0.8 * (deg / max_degree)) for deg in node_degrees]
        
        # Draw the graph
        nx.draw(self.tree, 
               pos=adjusted_positions,
               labels=labels,
               with_labels=label_nodes,
               node_color=node_colors,
               node_size=node_size,
               font_size=font_size,
               font_weight='bold',
               arrows=True,
               arrowsize=20,
               edge_color='gray',
               arrowstyle='->')
        
        # Add edge labels if requested
        if label_edges:
            self._draw_directional_edge_labels(adjusted_positions,
                                               font_size=max(6, font_size - 2),
                                               vertical_spacing=vertical_spacing)
        
        # Use custom title if provided, otherwise default
        if title is None:
            title = f"Meek STV Decision Tree\nElection: {self.m} winners from {self.num_cands} candidates"
        
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_tree(self, figsize=(15, 10), node_size=3000, font_size=8, label_nodes=False, 
                  label_edges=True, title=None, spacing_multiplier=3.0, vertical_spacing=None):
        """Backward compatible wrapper for older notebooks."""
        return self.plot_graph(figsize=figsize,
                               node_size=node_size,
                               font_size=font_size,
                               label_nodes=label_nodes,
                               label_edges=label_edges,
                               title=title,
                               spacing_multiplier=spacing_multiplier,
                               vertical_spacing=vertical_spacing)
    
    def get_stats(self):
        """Get statistics about the constructed tree"""
        stats = {
            'total_nodes': len(self.tree.nodes()),
            'total_edges': len(self.tree.edges()),
            'layers': len(self.nodes_by_layer),
            'nodes_per_layer': {layer: len(nodes) for layer, nodes in self.nodes_by_layer.items()}
        }
        return stats
    
    def analyze_paths(self):
        """Analyze all possible paths from root to leaves"""
        root = self.node_to_string({"winner_to_cand": [], "initial_losers": []})
        
        # Find all leaf nodes (nodes with no outgoing edges)
        leaves = [node for node in self.tree.nodes() if self.tree.out_degree(node) == 0]
        
        paths = []
        for leaf in leaves:
            try:
                path = nx.shortest_path(self.tree, root, leaf)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue
        
        return paths
    
    def print_analysis(self):
        """Print detailed analysis of the tree"""
        stats = self.get_stats()
        print("Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nCandidates: {self.candidates}")
        print(f"Target winners (m): {self.m}")
        print(f"Available node IDs: {self.list_nodes()}")
        
        paths = self.analyze_paths()
        print(f"\nTotal possible election paths: {len(paths)}")
        
        # Check if we have complete elections (paths reaching m winners)
        complete_elections = []
        for path in paths:
            leaf_node_str = path[-1]
            leaf_node = self.tree.nodes[leaf_node_str]
            if len(leaf_node['winners']) == self.m:
                complete_elections.append(path)
        
        print(f"Complete elections (with {self.m} winners): {len(complete_elections)}")
        
        if paths:
            print("\nSample paths (by Node ID):")
            for i, path in enumerate(paths[:3]):  # Show first 3 paths
                node_ids = [self.tree.nodes[node]['node_id'] for node in path]
                print(f"  Path {i+1}: {' → '.join(node_ids)}")
            
            if len(paths) > 3:
                print(f"  ... and {len(paths) - 3} more paths")
