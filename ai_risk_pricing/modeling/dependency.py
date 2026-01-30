"""
Dependency graph modeling for systemic AI risk propagation.

Uses NetworkX to model the AI supply chain and simulate loss propagation
from upstream failures to downstream enterprises.
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Node:
    """
    A node in the AI dependency graph.
    
    Represents an entity in the AI supply chain with exposure and
    connectivity characteristics that determine loss propagation.
    
    Actuarial interpretation:
        Each node is a potential loss source or transmission point.
        Exposure determines base loss potential.
        Dependency weight determines loss transmission from upstream.
        Criticality affects loss amplification.
    
    Attributes:
        name: Unique node identifier.
        node_type: Category (foundation_model, saas_provider, enterprise).
        exposure: Base exposure value in $M (maximum potential loss).
        dependency_weight: Proportion of upstream loss absorbed (0-1).
        criticality_score: System criticality multiplier (1.0 = normal).
    """
    
    name: str
    node_type: str
    exposure: float
    dependency_weight: float
    criticality_score: float = 1.0


class DependencyGraph:
    """
    NetworkX-based dependency graph for AI systemic risk.
    
    Models the AI supply chain as a directed graph where:
    - Foundation models are upstream (roots)
    - SaaS providers are middle tier
    - Enterprises are downstream (leaves)
    
    Losses propagate from upstream to downstream, amplified by
    concentration and criticality.
    
    Actuarial interpretation:
        In traditional cat modeling, dependency structures capture
        correlation in losses (e.g., multiple buildings in same
        earthquake zone). For AI, the dependency is functional:
        if a foundation model fails, all dependent services fail.
        
        The graph structure enables modeling of:
        - Concentration risk (many nodes depend on few providers)
        - Cascade failures (upstream failure propagates down)
        - Systemic risk (correlated losses across portfolio)
    """
    
    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self.graph = nx.DiGraph()
        self._nodes: dict[str, Node] = {}
    
    def add_node(self, node: Node) -> None:
        """
        Add a node to the dependency graph.
        
        Args:
            node: Node specification to add.
        """
        self._nodes[node.name] = node
        self.graph.add_node(
            node.name,
            node_type=node.node_type,
            exposure=node.exposure,
            dependency_weight=node.dependency_weight,
            criticality_score=node.criticality_score,
        )
    
    def add_dependency(self, upstream: str, downstream: str, weight: float = 1.0) -> None:
        """
        Add a directed dependency edge from upstream to downstream.
        
        The edge weight represents the strength of dependency (how much
        of upstream loss propagates to downstream).
        
        Args:
            upstream: Name of upstream node (loss source).
            downstream: Name of downstream node (loss receiver).
            weight: Edge weight for loss transmission (0-1).
        """
        if upstream not in self._nodes:
            raise ValueError(f"Unknown upstream node: {upstream}")
        if downstream not in self._nodes:
            raise ValueError(f"Unknown downstream node: {downstream}")
        
        self.graph.add_edge(upstream, downstream, weight=weight)
    
    def get_node(self, name: str) -> Node:
        """Get a node by name."""
        return self._nodes[name]
    
    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]
    
    def get_downstream_nodes(self, node_name: str) -> list[str]:
        """Get all nodes directly downstream of a given node."""
        return list(self.graph.successors(node_name))
    
    def get_upstream_nodes(self, node_name: str) -> list[str]:
        """Get all nodes directly upstream of a given node."""
        return list(self.graph.predecessors(node_name))
    
    def calculate_concentration_index(self) -> float:
        """
        Calculate Herfindahl-Hirschman Index for dependency concentration.
        
        HHI measures market concentration. For our graph, we calculate
        concentration based on the share of downstream dependencies
        held by each foundation model.
        
        Actuarial interpretation:
            High concentration (HHI close to 1) means losses are highly
            correlated - a single point of failure affects most of the
            portfolio. Low concentration means risks are diversified.
        
        Returns:
            Concentration index between 0 (perfect diversification)
            and 1 (perfect concentration).
        """
        foundation_models = self.get_nodes_by_type("foundation_model")
        if not foundation_models:
            return 0.0
        
        # Count downstream dependencies for each foundation model
        total_downstream = 0
        downstream_counts = []
        
        for fm in foundation_models:
            # Count all reachable downstream nodes
            reachable = len(nx.descendants(self.graph, fm.name))
            downstream_counts.append(reachable)
            total_downstream += reachable
        
        if total_downstream == 0:
            return 0.0
        
        # Calculate HHI (sum of squared market shares)
        shares = [count / total_downstream for count in downstream_counts]
        hhi = sum(s**2 for s in shares)
        
        return hhi
    
    def propagate_loss(
        self,
        root_node: str,
        root_loss: float,
        base_propagation: float = 0.65,
        concentration_exponent: float = 2.0,
        max_amplification: float = 5.0,
    ) -> dict[str, float]:
        """
        Propagate loss from a root node through the dependency graph.
        
        Loss transmission follows the graph structure, with amplification
        based on concentration risk. The propagation is NOT purely linear -
        concentration amplifies losses nonlinearly.
        
        Actuarial interpretation:
            When a foundation model fails, dependent services absorb
            a portion of the loss. High concentration (few providers
            serving many enterprises) amplifies total loss because:
            1. More nodes are affected
            2. Lack of alternatives increases business interruption
            3. Systemic nature prevents hedging
        
        The nonlinear amplification term (1 + HHI^exponent) captures
        the "super-linear" growth of systemic risk with concentration.
        
        Args:
            root_node: Name of the node where loss originates.
            root_loss: Initial loss amount at root node.
            base_propagation: Base proportion of loss transmitted (0-1).
            concentration_exponent: Exponent for concentration amplification.
            max_amplification: Cap on total amplification factor.
        
        Returns:
            Dictionary mapping node names to their loss amounts.
        """
        if root_node not in self._nodes:
            raise ValueError(f"Unknown root node: {root_node}")
        
        # Calculate concentration-based amplification
        concentration = self.calculate_concentration_index()
        amplification = min(
            1 + concentration**concentration_exponent,
            max_amplification,
        )
        
        # Initialize loss tracking
        node_losses: dict[str, float] = {root_node: root_loss}
        visited: set[str] = set()
        
        # BFS propagation through graph
        queue = [root_node]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            current_loss = node_losses.get(current, 0.0)
            current_node = self._nodes[current]
            
            # Propagate to downstream nodes
            for downstream in self.get_downstream_nodes(current):
                if downstream in visited:
                    continue
                
                downstream_node = self._nodes[downstream]
                edge_weight = self.graph[current][downstream].get("weight", 1.0)
                
                # Calculate propagated loss
                # Base propagation * edge weight * downstream dependency * amplification
                propagated = (
                    current_loss
                    * base_propagation
                    * edge_weight
                    * downstream_node.dependency_weight
                    * amplification
                )
                
                # Apply criticality multiplier
                propagated *= downstream_node.criticality_score
                
                # Cap at node's exposure
                propagated = min(propagated, downstream_node.exposure)
                
                # Accumulate loss at downstream node
                node_losses[downstream] = node_losses.get(downstream, 0.0) + propagated
                queue.append(downstream)
                breakpoint()
        
        return node_losses
    
    def total_propagated_loss(
        self,
        root_node: str,
        root_loss: float,
        **kwargs,
    ) -> float:
        """
        Calculate total loss across all nodes from a root failure.
        
        Convenience method that sums losses across the entire graph.
        
        Args:
            root_node: Name of the node where loss originates.
            root_loss: Initial loss amount at root node.
            **kwargs: Additional arguments passed to propagate_loss.
        
        Returns:
            Total loss summed across all affected nodes.
        """
        node_losses = self.propagate_loss(root_node, root_loss, **kwargs)
        return sum(node_losses.values())
    
    def total_exposure(self) -> float:
        """Calculate total exposure across all nodes."""
        return sum(node.exposure for node in self._nodes.values())
    
    @classmethod
    def build_sample_graph(cls) -> "DependencyGraph":
        """
        Build a sample AI supply chain dependency graph.
        
        Creates a realistic three-tier structure:
        - 2 Foundation models (high exposure, upstream)
        - 4 SaaS providers (medium exposure, middle tier)
        - 8 Enterprises (variable exposure, downstream)
        
        Returns:
            Populated dependency graph for simulation.
        """
        graph = cls()
        
        # Foundation Models (upstream)
        graph.add_node(Node(
            name="foundation_model_alpha",
            node_type="foundation_model",
            exposure=500.0,
            dependency_weight=1.0,
            criticality_score=2.0,
        ))
        graph.add_node(Node(
            name="foundation_model_beta",
            node_type="foundation_model",
            exposure=400.0,
            dependency_weight=1.0,
            criticality_score=1.8,
        ))
        
        # SaaS Providers (middle tier)
        saas_configs = [
            ("saas_provider_1", 150.0, 0.8, 1.5),
            ("saas_provider_2", 120.0, 0.7, 1.3),
            ("saas_provider_3", 100.0, 0.75, 1.4),
            ("saas_provider_4", 80.0, 0.65, 1.2),
        ]
        for name, exposure, dep_weight, crit in saas_configs:
            graph.add_node(Node(
                name=name,
                node_type="saas_provider",
                exposure=exposure,
                dependency_weight=dep_weight,
                criticality_score=crit,
            ))
        
        # Enterprises (downstream)
        enterprise_configs = [
            ("enterprise_1", 50.0, 0.6, 1.0),
            ("enterprise_2", 75.0, 0.7, 1.1),
            ("enterprise_3", 40.0, 0.5, 0.9),
            ("enterprise_4", 60.0, 0.65, 1.0),
            ("enterprise_5", 90.0, 0.8, 1.2),
            ("enterprise_6", 35.0, 0.55, 0.8),
            ("enterprise_7", 55.0, 0.6, 1.0),
            ("enterprise_8", 45.0, 0.5, 0.9),
        ]
        for name, exposure, dep_weight, crit in enterprise_configs:
            graph.add_node(Node(
                name=name,
                node_type="enterprise",
                exposure=exposure,
                dependency_weight=dep_weight,
                criticality_score=crit,
            ))
        
        # Dependencies: Foundation Models -> SaaS Providers
        # Model Alpha serves providers 1, 2, 3 (concentration risk)
        graph.add_dependency("foundation_model_alpha", "saas_provider_1", weight=0.9)
        graph.add_dependency("foundation_model_alpha", "saas_provider_2", weight=0.85)
        graph.add_dependency("foundation_model_alpha", "saas_provider_3", weight=0.8)
        
        # Model Beta serves providers 3, 4 (some overlap with Alpha)
        graph.add_dependency("foundation_model_beta", "saas_provider_3", weight=0.7)
        graph.add_dependency("foundation_model_beta", "saas_provider_4", weight=0.9)
        
        # Dependencies: SaaS Providers -> Enterprises
        graph.add_dependency("saas_provider_1", "enterprise_1", weight=0.8)
        graph.add_dependency("saas_provider_1", "enterprise_2", weight=0.7)
        graph.add_dependency("saas_provider_2", "enterprise_2", weight=0.6)
        graph.add_dependency("saas_provider_2", "enterprise_3", weight=0.75)
        graph.add_dependency("saas_provider_2", "enterprise_4", weight=0.8)
        graph.add_dependency("saas_provider_3", "enterprise_4", weight=0.5)
        graph.add_dependency("saas_provider_3", "enterprise_5", weight=0.85)
        graph.add_dependency("saas_provider_3", "enterprise_6", weight=0.7)
        graph.add_dependency("saas_provider_4", "enterprise_6", weight=0.6)
        graph.add_dependency("saas_provider_4", "enterprise_7", weight=0.8)
        graph.add_dependency("saas_provider_4", "enterprise_8", weight=0.75)
        
        return graph
