"""
INDRA REST API Client
Query biological knowledge for causal network priors.

INDRA API Documentation:
- REST API: http://api.indra.bio:8000 (general INDRA services)
- Database REST API: https://db.indra.bio (statement queries)
- Python Client: indra.sources.indra_db_rest.get_statements()

Recommended Usage:
  Use the official INDRA Python client for best reliability:

  import os
  os.environ['INDRA_DB_REST_URL'] = 'https://db.indra.bio'

  import indra.sources.indra_db_rest as idr
  processor = idr.get_statements(subject='EGFR', object='KRAS', limit=100)
  statements = processor.statements
"""

import os
import time
from typing import List, Dict, Tuple, Optional
import logging

# Set INDRA DB URL environment variable
os.environ['INDRA_DB_REST_URL'] = 'https://db.indra.bio'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndraClient:
    """
    Client for INDRA Database REST API.

    Uses the official INDRA Python library (indra.sources.indra_db_rest)
    which handles retries, timeouts, and API changes properly.

    Pattern based on working implementation from digitalme/indra_agent.
    """

    def __init__(self):
        self.cache = {}  # Cache queries to avoid redundant API calls

        # Import INDRA library
        try:
            import indra.sources.indra_db_rest as idr
            self.idr = idr
        except ImportError:
            logger.error("INDRA library not installed. Install with: pip install indra")
            raise
        
    def get_statements(self,
                      subject: Optional[str] = None,
                      object: Optional[str] = None,
                      stmt_type: Optional[str] = None,
                      limit: int = 100,
                      timeout: int = 30) -> List:
        """
        Query INDRA for statements involving specific agents.

        Uses the official INDRA Python client for reliability.

        Args:
            subject: Gene symbol for subject (e.g., "EGFR")
            object: Gene symbol for object (e.g., "KRAS")
            stmt_type: Type of statement (e.g., "Phosphorylation", "Activation")
            limit: Maximum statements to return
            timeout: Request timeout in seconds

        Returns:
            List of INDRA Statement objects
        """
        # Build cache key
        cache_key = f"{subject}_{object}_{stmt_type}_{limit}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]

        try:
            # Use official INDRA client
            processor = self.idr.get_statements(
                subject=subject,
                object=object,
                stmt_type=stmt_type,
                limit=limit,
                timeout=timeout
            )

            statements = processor.statements if processor else []

            # Cache result
            self.cache[cache_key] = statements

            logger.info(f"Retrieved {len(statements)} statements for {cache_key}")
            return statements

        except Exception as e:
            logger.error(f"INDRA API request failed: {e}")
            logger.warning("Returning empty list - INDRA may be unavailable")
            return []
            
    def get_regulation_type(self, gene1: str, gene2: str) -> Dict[str, float]:
        """
        Determine if gene1 regulates gene2 (and how).

        Returns:
            Dict with 'activates', 'inhibits', 'phosphorylates', etc. belief scores
        """
        regulation_types = {
            'activates': 0.0,
            'inhibits': 0.0,
            'phosphorylates': 0.0,
            'increases_amount': 0.0,
            'decreases_amount': 0.0,
        }

        # Query for all statement types
        all_stmts = self.get_statements(subject=gene1, object=gene2)

        for stmt in all_stmts:
            # INDRA Statement objects have a __class__.__name__ attribute
            stmt_type = stmt.__class__.__name__.lower()
            belief = getattr(stmt, 'belief', 0.0)

            if 'activation' in stmt_type:
                regulation_types['activates'] = max(regulation_types['activates'], belief)
            elif 'inhibition' in stmt_type:
                regulation_types['inhibits'] = max(regulation_types['inhibits'], belief)
            elif 'phosphorylation' in stmt_type:
                regulation_types['phosphorylates'] = max(regulation_types['phosphorylates'], belief)
            elif 'increaseamount' in stmt_type:
                regulation_types['increases_amount'] = max(regulation_types['increases_amount'], belief)
            elif 'decreaseamount' in stmt_type:
                regulation_types['decreases_amount'] = max(regulation_types['decreases_amount'], belief)

        return regulation_types
    
    def build_prior_network(self, genes: List[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Build prior network from INDRA knowledge.

        Args:
            genes: List of gene symbols

        Returns:
            Dict mapping (gene1, gene2) -> {'belief': max_belief, 'type': stmt_type}
        """
        network = {}

        logger.info(f"Building prior network for {len(genes)} genes...")

        for i, gene1 in enumerate(genes):
            for gene2 in genes:
                if gene1 == gene2:
                    continue

                # Get regulation evidence
                reg = self.get_regulation_type(gene1, gene2)

                # Take max belief across all types
                max_belief = max(reg.values())

                if max_belief > 0.0:
                    # Find which type had max belief
                    max_type = max(reg, key=reg.get)

                    network[(gene1, gene2)] = {
                        'belief': max_belief,
                        'type': max_type
                    }
                    logger.debug(f"{gene1} -> {gene2}: {max_type} (belief={max_belief:.3f})")

            logger.info(f"Processed {i+1}/{len(genes)} genes")
            time.sleep(0.1)  # Rate limiting

        logger.info(f"Built prior network with {len(network)} edges")
        return network
    
    def get_drug_targets(self, gene: str) -> List[Dict]:
        """
        Find drugs that target a specific gene.

        Args:
            gene: Gene symbol (e.g., "EGFR")

        Returns:
            List of dicts with 'drug_name' and 'mechanism'
        """
        # Query for inhibition statements where gene is object
        stmts = self.get_statements(object=gene, stmt_type="Inhibition")

        drugs = []
        for stmt in stmts:
            # INDRA Statement objects have .subj and .obj attributes
            if hasattr(stmt, 'subj') and stmt.subj:
                subj_name = stmt.subj.name

                # Filter for likely drug names (heuristic: lowercase, ends in -ib, -mab, etc.)
                if any(subj_name.lower().endswith(suffix) for suffix in ['ib', 'mab', 'nib', 'tinib']):
                    drugs.append({
                        'drug_name': subj_name,
                        'mechanism': f"Inhibits {gene}",
                        'belief': getattr(stmt, 'belief', 0.0)
                    })

        # Sort by belief
        drugs.sort(key=lambda x: x['belief'], reverse=True)

        return drugs

# Example usage
if __name__ == "__main__":
    client = IndraClient()
    
    # Test: Get EGFR -> KRAS statements
    stmts = client.get_statements(subject="EGFR", object="KRAS")
    print(f"Found {len(stmts)} EGFR -> KRAS statements")
    
    # Test: Build small network
    genes = ["EGFR", "KRAS", "BRAF", "MEK1", "ERK1"]
    network = client.build_prior_network(genes)
    print(f"Built network with {len(network)} edges")
    
    # Test: Get drugs targeting EGFR
    drugs = client.get_drug_targets("EGFR")
    print(f"Found {len(drugs)} drugs targeting EGFR")
    for drug in drugs[:5]:
        print(f"  {drug['drug_name']}: {drug['mechanism']} (belief={drug['belief']:.3f})")
