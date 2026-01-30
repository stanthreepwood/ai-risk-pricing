import numpy as np
from .company import Portfolio
from ..modeling.dependency import DependencyGraph, Node


class PortfolioAggregator:
    """
    Aggregates portfolio exposures
    
    company-level portfolio view from the node-level dependency graph
    used for loss propagation.
    
   In catastrophe modeling, we need to map insured entities to the physical/logical structure that determines loss correlation.
   For AI risk, companies are mapped to their position in the AI supply chain (foundation model user, SaaS customer, etc.).
    """
    
    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio
    
    def build_dependency_graph_from_portfolio(
        self,
        n_foundation_models: int = 2,
        n_saas_providers: int = 4,
    ) -> DependencyGraph:
        """
        Build a dependency graph incorporating portfolio companies.
        
        Creates a complete AI supply chain graph with:
        - Foundation model nodes (upstream risk sources)
        - SaaS provider nodes (middle tier)
        - Enterprise nodes derived from portfolio companies
        
        Portfolio companies are mapped to enterprise nodes based on
        their characteristics. The graph structure models realistic
        concentration patterns in AI infrastructure.
        
        Args:
            n_foundation_models: Number of foundation model providers.
            n_saas_providers: Number of SaaS AI service providers.
        
        Returns:
            Complete dependency graph for simulation.
        """
        graph = DependencyGraph()
        rng = np.random.default_rng(42)
        
        fm_nodes = []
        for i in range(n_foundation_models):
            node = Node(
                name=f"foundation_model_{i+1}",
                node_type="foundation_model",
                exposure=500.0 - i * 100,
                dependency_weight=1.0,
                criticality_score=2.0 - i * 0.2,
            )
            graph.add_node(node)
            fm_nodes.append(node)
        
        saas_nodes = []
        for i in range(n_saas_providers):
            node = Node(
                name=f"saas_provider_{i+1}",
                node_type="saas_provider",
                exposure=100.0 + rng.uniform(-30, 30),
                dependency_weight=0.6 + rng.uniform(0, 0.3),
                criticality_score=1.2 + rng.uniform(-0.2, 0.3),
            )
            graph.add_node(node)
            saas_nodes.append(node)
        
        enterprise_nodes = []
        for company in self.portfolio.companies:
            node = Node(
                name=f"enterprise_{company.name.replace(' ', '_').lower()}",
                node_type="enterprise",
                exposure=company.exposure,
                dependency_weight=company.ai_dependency_score,
                criticality_score=1.0 + company.risk_score * 0.5,
            )
            graph.add_node(node)
            enterprise_nodes.append(node)
            
        for i, saas in enumerate(saas_nodes):
            #TODO: unmock this
            # Primary foundation model dependency
            primary_fm = fm_nodes[0] if i < len(saas_nodes) * 0.6 else fm_nodes[min(1, len(fm_nodes)-1)]
            graph.add_dependency(
                primary_fm.name,
                saas.name,
                weight=0.8 + rng.uniform(0, 0.15),
            )
            
            # Some providers have secondary dependencies
            if rng.random() > 0.5 and len(fm_nodes) > 1:
                secondary_fm = fm_nodes[1] if primary_fm == fm_nodes[0] else fm_nodes[0]
                graph.add_dependency(
                    secondary_fm.name,
                    saas.name,
                    weight=0.3 + rng.uniform(0, 0.2),
                )
        
        #SaaS Providers -> Enterprises,  based on AI dependency
        for ent in enterprise_nodes:
            #higher AI dependency = more SaaS connections
            n_saas_deps = max(1, int(self._get_company_ai_dep(ent.name) * len(saas_nodes)))
            n_saas_deps = min(n_saas_deps, len(saas_nodes))
            
            #select SaaS providers (w toward larger ones)
            selected_saas = rng.choice(
                saas_nodes,
                size=n_saas_deps,
                replace=False,
            )
            
            for saas in selected_saas:
                graph.add_dependency(
                    saas.name,
                    ent.name,
                    weight=0.5 + rng.uniform(0, 0.4),
                )
        
        return graph
    
    def _get_company_ai_dep(self, node_name: str) -> float:
        for company in self.portfolio.companies:
            if company.name.replace(' ', '_').lower() in node_name:
                return company.ai_dependency_score
        return 0.5
    
    def calculate_portfolio_loss_share(
        self,
        node_losses: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate loss share for each portfolio company from node losses
        """
        company_losses: dict[str, float] = {}
        
        for company in self.portfolio.companies:
            node_name = f"enterprise_{company.name.replace(' ', '_').lower()}"
            company_losses[company.name] = node_losses.get(node_name, 0.0)
        
        return company_losses
    
    def portfolio_loss_allocation(
        self,
        total_loss: float,
        allocation_method: str = "exposure_weighted",
    ) -> dict[str, float]:
        """
        Allocate total portfolio loss to individual companies.
        
        Used when we have a portfolio-level loss and need to attribute
        it back to individual companies for analysis.
        
        Actuarial interpretation:
            Loss allocation is needed for per-policy pricing and for
            understanding which parts of the portfolio drive total loss.
            Exposure-weighted allocation is standard in cat modeling.
        
        Args:
            total_loss: Total portfolio loss to allocate.
            allocation_method: How to distribute loss ("exposure_weighted" or "equal").
        
        Returns:
            Dictionary mapping company names to allocated losses.
        """
        if allocation_method == "equal":
            per_company = total_loss / len(self.portfolio.companies)
            return {c.name: per_company for c in self.portfolio.companies}
        
        elif allocation_method == "exposure_weighted":
            total_exposure = self.portfolio.total_exposure
            if total_exposure == 0:
                return {c.name: 0.0 for c in self.portfolio.companies}
            
            return {
                c.name: total_loss * (c.exposure / total_exposure)
                for c in self.portfolio.companies
            }
        
        else:
            raise ValueError(f"Unknown allocation method: {allocation_method}")
