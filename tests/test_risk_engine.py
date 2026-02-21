from src.risk_engine.rules import RiskEngine

def test_helmet_missing():
    engine = RiskEngine()
    risks = engine.evaluate(["person"])
    assert "Helmet Missing" in risks