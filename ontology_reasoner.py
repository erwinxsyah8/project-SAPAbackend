from rdflib import Graph, Namespace
from pathlib import Path

OCEAN = Namespace("http://example.org/ocean#")

class OceanReasoner:
    def __init__(self):
        ontology_path = Path(__file__).parent / "ontology" / "ocean.owl"
        self.graph = Graph()
        self.graph.parse(ontology_path, format="turtle")

    def infer_behaviors(self, level: str):
        """
        level: Low | Medium | High
        """
        query = f"""
        SELECT ?behavior WHERE {{
            ocean:{level} ocean:inducesBehavior ?behavior .
        }}
        """
        results = self.graph.query(query, initNs={"ocean": OCEAN})
        return [str(r[0]).split("#")[-1] for r in results]

    def build_semantic_explanation(self, trait: str, score: float, confidence: float, uncertainty: float):
        """
        Returns ontology-driven explanation
        """
        if score <= 0.33:
            level = "Low"
        elif score <= 0.66:
            level = "Medium"
        else:
            level = "High"

        behaviors = self.infer_behaviors(level)

        reliability = "High"
        if uncertainty > 0.02:
            reliability = "Medium"
        if uncertainty > 0.05:
            reliability = "Low"

        return {
            "trait": trait,
            "score": round(score, 3),
            "level": level,
            "confidence": round(confidence, 3),
            "uncertainty": round(uncertainty, 4),
            "reliability": reliability,
            "behaviors": behaviors,
        }
