# TBD
#from pathlib import Path
#from tkinter import EventType
#from ai_risk_pricing.scenario.schema import NodeLayer, Scenario
#from load_regulation_data import load_risks
#from quantify_regulation_data import aggregate_frequency, aggregate_severity_distribution


#if __name__ == "__main__":
#    ai_act_risks = load_risks(f"{Path(__file__).parent}/data/ai_act.yaml")
#    for risk in ai_act_risks["ai_act_risk_taxonomy"]:
#        frequency = aggregate_frequency([risk])
#        severity_distribution = aggregate_severity_distribution([risk])
#        scenario = Scenario(
#            name=risk["name"],
#            event_type=EventType.MODEL_COLLAPSE,
#            trigger=risk["trigger"],
#            propagation_vector=risk["propagation_vector"],
#            affected_nodes=[NodeLayer(risk["layer"])],
#            base_frequency=frequency,
#            severity_distribution=severity_distribution,
#            severity_materialization=None,
#            legal_frameworks=risk["legal_frameworks"],
#            tail_multiplier=1.0,
#            capability_threshold=0.7,
#            threshold_multiplier=3.0,
#            metadata={},
#        )
#        breakpoint()
#        print(frequency)
#        print(severity_distribution)