# =========================================================
# ENTERPRISE INTELLIGENCE LAYER
# =========================================================

RISK_CATEGORY_MAPPING = {
    "critical": "A",
    "high": "B",
    "medium": "C",
    "low": "C",
}


DEPARTMENT_MAPPING = {

    "surface_crack": "Civil Engineering",

    "corrosion": "Maintenance",

    "water_leakage": "Civil Engineering",

    "rebar_exposure": "Construction Quality",

    "material_damage": "Construction Quality",

    "poor_housekeeping": "Site Operations",

    "ppe_non_compliance": "Safety Department",
}

DEPARTMENT_EXECUTIVE_OWNER = {

    "Civil Engineering": "Engineering Manager",

    "Maintenance": "Maintenance Manager",

    "Construction Quality": "Quality Assurance Manager",

    "Site Operations": "Site Operations Manager",

    "Safety Department": "HSE Manager",
}

CLOSURE_TIMELINE = {

    "critical": "Immediate (0-24 Hours)",

    "high": "7 Days",

    "medium": "30 Days",

    "low": "Next Inspection Cycle",
}

AUDIT_PRIORITY = {

    "critical": "Immediate Escalation",

    "high": "Management Review",

    "medium": "Supervisor Review",

    "low": "Routine Monitoring",
}

AUDIT_EXPOSURE_LIBRARY = {

    "critical": "High probability of major audit non-conformance.",

    "high": "Potential significant audit observation.",

    "medium": "Moderate audit exposure requiring corrective action.",

    "low": "Limited audit exposure.",
}

CONSEQUENCE_LIBRARY = {

    "surface_crack": [
        "Progressive structural deterioration",
        "Reduced load-bearing performance",
        "Increased maintenance expenditure",
        "Potential structural instability",
    ],

    "corrosion": [
        "Material section loss",
        "Reduced structural reliability",
        "Accelerated infrastructure degradation",
        "Increased lifecycle repair costs",
    ],

    "water_leakage": [
        "Material deterioration",
        "Electrical hazard exposure",
        "Mould and moisture damage",
        "Reduced infrastructure lifespan",
    ],

    "rebar_exposure": [
        "Accelerated reinforcement corrosion",
        "Reduced structural durability",
        "Loss of concrete protection",
        "Potential structural weakening",
    ],

    "material_damage": [
        "Reduced operational reliability",
        "Premature material failure",
        "Increased maintenance burden",
        "Potential safety exposure",
    ],

    "poor_housekeeping": [
        "Trip and fall injuries",
        "Restricted worker movement",
        "Emergency evacuation obstruction",
        "Reduced operational efficiency",
    ],

    "ppe_non_compliance": [
        "Worker injury exposure",
        "Regulatory non-compliance",
        "Lost-time incidents",
        "Increased liability risk",
    ],
}

MANAGEMENT_IMPACT_LIBRARY = {

    "surface_crack":
        "Recurring structural cracking patterns may indicate deficiencies "
        "in construction quality control, curing management, or load "
        "management practices.",

    "corrosion":
        "Corrosion findings indicate potential weaknesses in asset "
        "preservation strategy and long-term infrastructure maintenance "
        "planning.",

    "water_leakage":
        "Water ingress observations indicate potential deficiencies in "
        "waterproofing management and drainage performance controls.",

    "rebar_exposure":
        "Reinforcement exposure indicates potential shortcomings in "
        "construction quality assurance and concrete protection controls.",

    "material_damage":
        "Material damage findings may indicate deficiencies in material "
        "handling, storage management, or operational protection "
        "procedures.",

    "poor_housekeeping":
        "Recurring housekeeping deficiencies indicate weak supervisory "
        "enforcement, reduced workforce discipline, and increased audit "
        "exposure.",

    "ppe_non_compliance":
        "PPE non-compliance findings indicate weaknesses in safety "
        "culture, workforce accountability, and supervisory enforcement "
        "practices.",
}

DEFAULT_MANAGEMENT_IMPACT = (
    "The identified finding may affect operational performance and "
    "compliance outcomes."
)

EXECUTIVE_RECOMMENDATIONS = {

    "surface_crack":
        "Implement structural condition monitoring and engineering review "
        "procedures.",

    "corrosion":
        "Strengthen asset preservation strategy and protective coating "
        "management.",

    "water_leakage":
        "Improve waterproofing inspection and drainage maintenance controls.",

    "rebar_exposure":
        "Enhance construction quality verification and reinforcement "
        "protection controls.",

    "material_damage":
        "Improve material handling, storage, and inspection procedures.",

    "poor_housekeeping":
        "Implement site-wide housekeeping audits and supervisory "
        "accountability.",

    "ppe_non_compliance":
        "Strengthen safety culture initiatives and PPE enforcement "
        "programs.",
}

DEFAULT_EXECUTIVE_RECOMMENDATION = (
    "Maintain established quality assurance controls."
)


GUIDELINES = {

    # =====================================================
    # CRACKS
    # =====================================================

    "surface_crack": {

        "observation": (
            "Visible surface cracking patterns were identified on the "
            "inspected structural component, indicating material stress "
            "concentration or progressive surface degradation."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis indicates probable structural "
                "overloading beyond original design parameters, typically "
                "compounded by inadequate curing practices and cumulative "
                "thermal movement stress that progressively weakened the "
                "affected structural member."
            ),

            "high": (
                "Contributing root causes likely include sustained thermal "
                "movement (expansion and contraction cycling) interacting "
                "with insufficiently cured concrete, accelerating crack "
                "propagation under operational loading."
            ),

            "medium": (
                "The primary contributing factor is typically improper "
                "curing during the early concrete hydration period, "
                "exacerbated by thermal movement arising from environmental "
                "temperature fluctuations."
            ),

            "low": (
                "Improper curing practices during the initial concrete "
                "placement and hydration period are the most probable root "
                "cause, resulting in early-stage shrinkage-related surface "
                "cracking."
            ),
        },

        "risk": {

            "critical": (
                "Severe crack propagation presents an immediate structural "
                "failure risk. Load-bearing capacity may be critically "
                "compromised, posing direct safety hazards to personnel "
                "and surrounding infrastructure."
            ),

            "high": (
                "Significant crack progression may accelerate structural "
                "deterioration and reduce load-bearing performance if "
                "corrective action is not taken promptly."
            ),

            "medium": (
                "Moderate crack activity may reduce long-term structural "
                "durability and increase maintenance requirements if "
                "left unmonitored under environmental or operational loading."
            ),

            "low": (
                "Minor surface cracking presents limited immediate risk "
                "but should be monitored to prevent progression under "
                "sustained loading or environmental exposure."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately suspend structural loading operations in the "
                "affected zone. Commission an emergency structural engineering "
                "assessment and apply interim stabilisation measures. Formal "
                "written engineering clearance is mandatory before resuming "
                "any operations in the affected area."
            ),

            "high": (
                "Initiate accelerated structural assessment within 48 hours. "
                "Conduct crack mapping, measure width and progression rate, "
                "and apply appropriate sealant treatment. Restrict operational "
                "loading in affected zones pending engineering review."
            ),

            "medium": (
                "Schedule a detailed crack inspection within 30 days. Document "
                "crack dimensions, orientation, and progression rate. Apply "
                "corrective sealant treatment based on engineering evaluation "
                "findings."
            ),

            "low": (
                "Record crack location and visual characteristics during the "
                "next scheduled inspection cycle. Apply preventive sealant "
                "treatment if crack width exceeds 0.1 mm."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement continuous structural monitoring with defined load "
                "restriction protocols. Establish regular engineer-supervised "
                "inspection intervals and maintain detailed crack progression "
                "records."
            ),

            "high": (
                "Establish a structured crack monitoring programme with defined "
                "inspection intervals. Implement load management controls and "
                "environmental exposure mitigation measures."
            ),

            "medium": (
                "Conduct periodic structural inspections and maintain crack "
                "progression records. Monitor environmental conditions that may "
                "accelerate surface degradation between inspection cycles."
            ),

            "low": (
                "Include crack monitoring in routine inspection procedures. "
                "Maintain environmental exposure records and apply preventive "
                "protective coatings as conditions warrant."
            ),
        },

        "best_practice": (
            "Maintain controlled curing conditions during construction, "
            "implement structural load monitoring, and conduct regular material "
            "quality verification throughout the infrastructure lifecycle."
        ),

        "guideline_reference": (
            "Infrastructure Quality Standard — Structural Integrity Monitoring Protocol"
        ),
    },

    # =====================================================
    # CORROSION
    # =====================================================

    "corrosion": {

        "observation": (
            "Corrosion-related surface deterioration and material oxidation "
            "indicators were visually detected during inspection analysis."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis points to long-term coating failure that "
                "has permitted sustained moisture ingress into the substrate, "
                "allowing oxidation to progress unchecked to a critical "
                "material loss stage."
            ),

            "high": (
                "Protective coating failure combined with persistent moisture "
                "ingress is the likely root cause, allowing corrosion to "
                "progress significantly before detection."
            ),

            "medium": (
                "Moisture ingress through compromised or aging protective "
                "coatings is the probable root cause driving continued "
                "corrosion activity."
            ),

            "low": (
                "Early-stage moisture ingress, likely due to minor coating "
                "degradation, is the most probable root cause of the observed "
                "corrosion indicators."
            ),
        },

        "risk": {

            "critical": (
                "Advanced corrosion may have critically reduced material "
                "cross-section and load-bearing capacity, presenting immediate "
                "structural failure and personnel safety hazard risks."
            ),

            "high": (
                "Significant corrosion activity is actively weakening material "
                "performance and may accelerate infrastructure degradation if "
                "corrective action is delayed beyond the current operational cycle."
            ),

            "medium": (
                "Progressive corrosion may reduce material durability and "
                "operational lifespan if protective treatment is not applied "
                "within a reasonable timeframe."
            ),

            "low": (
                "Early-stage corrosion presents limited immediate structural "
                "risk but should be treated promptly to prevent accelerated "
                "surface deterioration."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately restrict operational loading on affected components. "
                "Commission emergency structural assessment to evaluate residual "
                "material capacity. Replace or structurally reinforce severely "
                "degraded sections before resuming any operational activities."
            ),

            "high": (
                "Remove corrosion deposits from affected surfaces and assess "
                "residual material thickness within 48 hours. Apply industrial-grade "
                "corrosion protection treatment and evaluate the need for full "
                "component replacement."
            ),

            "medium": (
                "Clean affected surfaces, assess corrosion depth and areal coverage, "
                "and apply an appropriate protective coating system within 30 days. "
                "Schedule a follow-up inspection to verify treatment effectiveness."
            ),

            "low": (
                "Apply preventive anti-corrosion treatment to affected areas during "
                "the next scheduled maintenance cycle. Document the location and "
                "extent of corrosion activity for trend monitoring."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement continuous corrosion monitoring using defined inspection "
                "protocols and environmental controls. Establish a mandatory "
                "protective coating maintenance schedule with engineer sign-off."
            ),

            "high": (
                "Establish a structured corrosion inspection programme with defined "
                "intervals. Apply protective coating systems and implement "
                "environmental exposure controls in affected zones."
            ),

            "medium": (
                "Schedule periodic corrosion inspections and maintain protective "
                "coating condition records. Monitor humidity and environmental "
                "exposure in high-risk areas."
            ),

            "low": (
                "Include corrosion checks in routine inspection procedures. Maintain "
                "protective coatings and address any coating damage promptly to "
                "prevent moisture ingress."
            ),
        },

        "best_practice": (
            "Specify corrosion-resistant materials for exposed infrastructure, "
            "maintain comprehensive protective coating programmes, and control "
            "environmental exposure through effective drainage and ventilation "
            "management."
        ),

        "guideline_reference": (
            "Industrial Asset Preservation & Corrosion Prevention Standard"
        ),
    },

    # =====================================================
    # WATER LEAKAGE
    # =====================================================

    "water_leakage": {

        "observation": (
            "Moisture intrusion and active or residual water leakage indicators "
            "were identified within the inspected construction area."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis indicates a combined failure of the "
                "waterproofing membrane system and underlying drainage "
                "deficiency, allowing uncontrolled water accumulation and "
                "ingress into the structure."
            ),

            "high": (
                "Waterproofing system failure, compounded by inadequate "
                "drainage capacity in the affected zone, is the probable root "
                "cause of the significant moisture infiltration observed."
            ),

            "medium": (
                "A developing waterproofing failure, potentially aggravated "
                "by drainage deficiency in the surrounding area, is the "
                "likely root cause of the persistent leakage."
            ),

            "low": (
                "Minor waterproofing degradation, possibly linked to early "
                "drainage deficiency, is the probable root cause of the "
                "observed moisture indicators."
            ),
        },

        "risk": {

            "critical": (
                "Uncontrolled water ingress presents an immediate risk of "
                "structural damage, potential electrical hazard, and accelerated "
                "material deterioration. Continued unaddressed exposure may lead "
                "to rapid infrastructure failure."
            ),

            "high": (
                "Significant moisture infiltration is actively degrading structural "
                "components and may compromise waterproofing system integrity across "
                "a wider area if not addressed urgently."
            ),

            "medium": (
                "Persistent water leakage may progressively deteriorate structural "
                "materials, introduce mould risk, and reduce long-term infrastructure "
                "reliability."
            ),

            "low": (
                "Minor moisture indicators suggest early-stage waterproofing "
                "compromise. Monitoring and preventive treatment are recommended "
                "to prevent escalation."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately isolate the affected zone and identify the primary "
                "leakage source. Commission emergency waterproofing repair and "
                "assess structural components for water damage. Restrict personnel "
                "access until structural safety is confirmed by qualified engineering "
                "review."
            ),

            "high": (
                "Identify and repair the primary leakage source within 48 hours. "
                "Restore waterproofing membrane integrity and assess structural "
                "components for moisture-related damage. Verify drainage system "
                "performance across the affected zone."
            ),

            "medium": (
                "Identify the leakage source and schedule waterproofing repair "
                "within 30 days. Apply interim waterproofing treatment to prevent "
                "further ingress and monitor affected areas for moisture progression."
            ),

            "low": (
                "Inspect waterproofing joints and drainage channels for minor "
                "defects during the next maintenance cycle. Apply targeted sealant "
                "treatment and monitor affected areas at subsequent inspections."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement continuous moisture monitoring in the affected zone. "
                "Establish mandatory waterproofing inspection schedules and drainage "
                "system maintenance protocols with defined response thresholds."
            ),

            "high": (
                "Establish a structured waterproofing inspection programme. Inspect "
                "drainage systems at defined intervals and maintain records of "
                "moisture-prone areas across the site."
            ),

            "medium": (
                "Conduct periodic waterproofing condition assessments and drainage "
                "inspections. Monitor high-risk zones during and after wet weather "
                "conditions."
            ),

            "low": (
                "Include waterproofing condition checks in routine inspection "
                "procedures. Maintain drainage clearance and inspect sealant "
                "integrity at regular intervals."
            ),
        },

        "best_practice": (
            "Implement effective site drainage design, conduct pre-handover "
            "waterproofing validation, and maintain a proactive moisture monitoring "
            "programme throughout the infrastructure lifecycle."
        ),

        "guideline_reference": (
            "Construction Waterproofing & Drainage Compliance Framework"
        ),
    },

    # =====================================================
    # REBAR EXPOSURE
    # =====================================================

    "rebar_exposure": {

        "observation": (
            "Reinforcement exposure indicators were identified, suggesting "
            "concrete surface degradation, cover loss, or insufficient "
            "protective coverage during construction."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis indicates a critical loss of concrete "
                "cover, likely resulting from incorrect spacer or chair "
                "placement during construction, compounded by long-term "
                "carbonation-driven cover deterioration."
            ),

            "high": (
                "Insufficient concrete cover depth at the time of placement, "
                "combined with progressive carbonation of the surrounding "
                "concrete, is the probable root cause of the significant "
                "reinforcement exposure."
            ),

            "medium": (
                "Inadequate reinforcement positioning or insufficient cover "
                "depth during construction is the likely root cause, with "
                "concrete surface deterioration over time exposing the "
                "embedded steel."
            ),

            "low": (
                "Minor cover deficiency, potentially arising from formwork "
                "displacement or spacer misplacement during placement, is "
                "the probable root cause of the localised exposure."
            ),
        },

        "risk": {

            "critical": (
                "Severely exposed reinforcement is actively corroding, critically "
                "undermining structural integrity and load transfer capacity. "
                "Immediate load restriction and engineering intervention are "
                "required to prevent structural failure."
            ),

            "high": (
                "Significant rebar exposure is accelerating corrosion propagation "
                "and reducing both structural capacity and long-term concrete "
                "durability across the affected zone."
            ),

            "medium": (
                "Partial reinforcement exposure may allow progressive corrosion "
                "and reduce the structural protection provided by the concrete "
                "cover if left unaddressed."
            ),

            "low": (
                "Minor concrete cover deficiency presents limited immediate risk "
                "but should be addressed to prevent moisture-driven corrosion "
                "over the service life of the structure."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately restrict structural loading in the affected area. "
                "Commission emergency structural assessment to evaluate "
                "reinforcement condition and residual load capacity. Apply "
                "corrosion inhibitor treatment and restore concrete cover to "
                "code-compliant depth before resuming operations."
            ),

            "high": (
                "Apply corrosion inhibitor treatment to exposed reinforcement "
                "and restore concrete cover to the required protective depth "
                "within 14 days. Assess the extent of corrosion penetration "
                "and associated structural impact."
            ),

            "medium": (
                "Schedule concrete repair within 30 days. Clean exposed rebar, "
                "apply anti-corrosion primer, and restore concrete cover using "
                "appropriate repair mortar to the specified minimum depth."
            ),

            "low": (
                "Apply preventive concrete sealant to vulnerable areas during the "
                "next maintenance cycle. Document exposure location and extent "
                "for monitoring at subsequent inspections."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement continuous structural monitoring in the affected zone. "
                "Establish mandatory reinforcement cover inspection protocols and "
                "enforce cover depth verification at each stage of future concrete "
                "placement operations."
            ),

            "high": (
                "Establish a structured rebar cover inspection programme. Implement "
                "cover depth verification controls and concrete quality checks "
                "during all future concrete placement operations."
            ),

            "medium": (
                "Conduct periodic concrete cover assessments and maintain inspection "
                "records. Monitor exposed areas for corrosion progression during "
                "routine inspection cycles."
            ),

            "low": (
                "Include rebar cover checks in routine inspection procedures. "
                "Verify concrete cover depth compliance during future construction "
                "quality control operations."
            ),
        },

        "best_practice": (
            "Enforce minimum concrete cover requirements per design specifications, "
            "use properly positioned and tied reinforcement, conduct cover depth "
            "verification before and after concrete placement, and apply controlled "
            "curing procedures."
        ),

        "guideline_reference": (
            "Reinforced Concrete Protection & Durability Standard"
        ),
    },

    # =====================================================
    # MATERIAL DAMAGE
    # =====================================================

    "material_damage": {

        "observation": (
            "Physical material degradation or localised construction surface "
            "damage was identified during visual inspection analysis."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis suggests severe mechanical impact or "
                "operational overload exceeding material design tolerances, "
                "potentially compounded by pre-existing material quality "
                "deficiencies."
            ),

            "high": (
                "Repeated mechanical stress or impact loading beyond the "
                "material's design tolerance, combined with possible material "
                "quality variance, is the likely root cause of the significant "
                "deterioration."
            ),

            "medium": (
                "Moderate mechanical or environmental stress acting on "
                "materials with possible underlying quality or handling "
                "deficiencies is the probable root cause."
            ),

            "low": (
                "Minor incidental impact or handling-related stress is the "
                "most probable root cause of the observed surface damage."
            ),
        },

        "risk": {

            "critical": (
                "Severe material damage may have critically compromised structural "
                "or operational integrity, posing immediate safety risks and "
                "requiring urgent engineering assessment before continued use."
            ),

            "high": (
                "Significant material deterioration is reducing infrastructure "
                "performance and may accelerate further degradation if corrective "
                "action is not initiated within the current operational period."
            ),

            "medium": (
                "Moderate material damage may negatively affect operational "
                "reliability and increase long-term maintenance requirements "
                "if not addressed within a reasonable timeframe."
            ),

            "low": (
                "Minor material surface damage presents limited operational impact "
                "but should be repaired to prevent progression under environmental "
                "or mechanical loading."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately restrict operational use of the affected area. "
                "Commission engineering assessment to determine the extent of damage "
                "and structural implications. Replace or reinforce critically damaged "
                "materials under qualified supervision before resuming operations."
            ),

            "high": (
                "Assess the full extent of material damage and initiate accelerated "
                "repair or replacement procedures within 14 days. Engineering "
                "validation is required before resuming full operational loading "
                "in the affected zone."
            ),

            "medium": (
                "Schedule detailed inspection and repair of damaged materials within "
                "30 days. Apply appropriate repair procedures based on material "
                "type and damage extent as assessed by qualified personnel."
            ),

            "low": (
                "Address surface damage during the next scheduled maintenance cycle. "
                "Document damage location and extent and apply appropriate protective "
                "treatment to prevent further deterioration."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement enhanced material integrity monitoring and operational "
                "load management controls. Review material specifications and "
                "handling procedures to identify and eliminate root causes of "
                "recurring damage."
            ),

            "high": (
                "Establish periodic material integrity inspections with defined "
                "acceptance criteria. Review handling and operational procedures "
                "to minimise future damage risk."
            ),

            "medium": (
                "Conduct regular material condition assessments and maintain "
                "damage records. Implement controlled handling procedures for "
                "materials identified as vulnerable."
            ),

            "low": (
                "Include material condition checks in routine inspection procedures. "
                "Apply protective coatings or surface treatments to minimise "
                "environmental damage exposure."
            ),
        },

        "best_practice": (
            "Specify certified construction materials with documented quality test "
            "reports, implement controlled material handling and storage procedures, "
            "and conduct pre-installation quality verification inspections."
        ),

        "guideline_reference": (
            "Construction Material Integrity Assurance Framework"
        ),
    },

    # =====================================================
    # SITE HOUSEKEEPING
    # =====================================================

    "poor_housekeeping": {

        "observation": (
            "Construction housekeeping deficiencies and disorganised worksite "
            "conditions were identified during the inspection."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis points to a systemic breakdown in site "
                "supervision and waste management protocols, reflecting an "
                "absence of enforced housekeeping standards across the site."
            ),

            "high": (
                "Inadequate supervisory enforcement of housekeeping standards, "
                "combined with insufficient designated storage and disposal "
                "arrangements, is the probable root cause of the significant "
                "deficiencies observed."
            ),

            "medium": (
                "Inconsistent enforcement of housekeeping procedures and "
                "unclear designation of storage and disposal areas are the "
                "likely root causes of the moderate shortfalls identified."
            ),

            "low": (
                "Minor lapses in routine housekeeping discipline, likely due "
                "to momentary oversight rather than systemic failure, are the "
                "probable root cause."
            ),
        },

        "risk": {

            "critical": (
                "Severely disorganised worksite conditions are creating immediate "
                "safety hazards including trip, fall, and fire risks, indicating "
                "a significant breakdown in site management and safety controls."
            ),

            "high": (
                "Significant housekeeping deficiencies are elevating worker injury "
                "risk and may indicate broader failures in operational discipline "
                "and safety management."
            ),

            "medium": (
                "Moderate housekeeping shortfalls are creating unnecessary "
                "operational hazards and reducing site efficiency and compliance "
                "standards."
            ),

            "low": (
                "Minor housekeeping issues were identified. Prompt attention will "
                "prevent escalation and maintain operational compliance standards."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately halt non-essential operations and clear all identified "
                "safety hazards. Conduct an emergency site safety review and "
                "implement mandatory housekeeping enforcement before work resumes. "
                "Formal site manager escalation and corrective action reporting "
                "are required."
            ),

            "high": (
                "Clear debris, reorganise all work zones, and restore safe "
                "operational pathways within 24 hours. Conduct a formal safety "
                "briefing with all site personnel and implement immediate "
                "housekeeping enforcement."
            ),

            "medium": (
                "Organise and clear affected work areas within 48 hours. Designate "
                "material storage zones and restore safe access pathways. Reinforce "
                "housekeeping expectations with all site personnel."
            ),

            "low": (
                "Address housekeeping deficiencies during the current or next "
                "work shift. Remind site personnel of housekeeping requirements "
                "and the location of designated storage and disposal areas."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement mandatory daily housekeeping audits with defined "
                "accountability. Establish clear housekeeping standards and "
                "formal disciplinary procedures for persistent non-compliance."
            ),

            "high": (
                "Establish structured daily housekeeping checks and assign clear "
                "supervisory responsibility for site cleanliness. Implement "
                "workforce awareness training for housekeeping compliance."
            ),

            "medium": (
                "Conduct regular housekeeping inspections at defined intervals. "
                "Display housekeeping standards in prominent locations and "
                "reinforce expectations through scheduled toolbox talks."
            ),

            "low": (
                "Include housekeeping checks in daily site inspections. Maintain "
                "clearly marked material storage and waste disposal areas "
                "throughout the site."
            ),
        },

        "best_practice": (
            "Maintain enterprise housekeeping standards with clearly designated "
            "material storage, waste disposal, and access pathways. Conduct daily "
            "site inspections and enforce housekeeping accountability at all levels "
            "of site management."
        ),

        "guideline_reference": (
            "Construction Site Operational Discipline & Housekeeping Standard"
        ),
    },

    # =====================================================
    # PPE NON-COMPLIANCE
    # =====================================================

    "ppe_non_compliance": {

        "observation": (
            "Personal protective equipment non-compliance indicators were "
            "identified, with workers observed operating without required "
            "safety equipment in the inspected area."
        ),

        "root_cause": {

            "critical": (
                "Root cause analysis indicates a systemic failure in safety "
                "culture and supervisory enforcement, reflecting inadequate "
                "accountability structures for PPE compliance across the site."
            ),

            "high": (
                "Insufficient supervisory enforcement and inadequate worker "
                "safety awareness are the probable root causes of the "
                "significant non-compliance observed."
            ),

            "medium": (
                "Inconsistent supervisory enforcement of PPE requirements, "
                "potentially combined with gaps in worker safety awareness, "
                "is the likely root cause."
            ),

            "low": (
                "An isolated lapse in individual compliance, rather than a "
                "systemic enforcement gap, is the probable root cause of the "
                "identified non-compliance."
            ),
        },

        "risk": {

            "critical": (
                "Multiple or repeated PPE non-compliance events indicate a "
                "systemic safety culture failure. Worker injury risk is critically "
                "elevated and regulatory enforcement penalties are likely to apply."
            ),

            "high": (
                "Significant PPE non-compliance is creating serious worker injury "
                "exposure and indicates inadequate safety enforcement and supervision "
                "within the affected work zone."
            ),

            "medium": (
                "PPE non-compliance in the affected area is elevating worker injury "
                "risk and may compromise the site's overall safety compliance record."
            ),

            "low": (
                "Isolated PPE non-compliance was identified. Prompt corrective "
                "instruction will prevent escalation and maintain the site's "
                "safety compliance standards."
            ),
        },

        "corrective_action": {

            "critical": (
                "Immediately suspend work in the affected zone. Conduct an emergency "
                "safety inspection and brief all workers on mandatory PPE requirements "
                "before work resumes. Report all non-compliance incidents formally to "
                "safety management for corrective action and regulatory notification "
                "as required."
            ),

            "high": (
                "Issue immediate safety instruction to all non-compliant workers and "
                "remove them from work activities until compliant PPE is confirmed. "
                "Conduct a formal safety briefing with all site personnel within "
                "24 hours."
            ),

            "medium": (
                "Issue direct corrective instruction to workers identified without "
                "required PPE. Conduct a team safety reminder and verify PPE "
                "availability and adequacy across the affected work zone."
            ),

            "low": (
                "Provide immediate PPE instruction to the identified individual. "
                "Verify PPE availability, condition, and suitability for the "
                "specific work being performed."
            ),
        },

        "preventive_action": {

            "critical": (
                "Implement continuous PPE compliance monitoring through supervisory "
                "checks and site surveillance. Establish formal disciplinary "
                "procedures for repeat non-compliance and conduct regular safety "
                "culture assessments."
            ),

            "high": (
                "Establish mandatory PPE compliance checks at site entry points and "
                "work zone boundaries. Implement supervisor accountability for PPE "
                "enforcement within their areas of responsibility."
            ),

            "medium": (
                "Conduct regular PPE compliance audits and include compliance results "
                "in site safety reporting. Implement PPE awareness training as part "
                "of site induction and refresher procedures."
            ),

            "low": (
                "Include PPE compliance in daily site inspection checklists. Ensure "
                "adequate PPE supplies are accessible at site entry points and "
                "across all active work zones."
            ),
        },

        "best_practice": (
            "Enforce mandatory PPE requirements at all site access points, conduct "
            "regular compliance audits, provide regular safety induction and "
            "refresher training, and maintain a safety-first culture with clear "
            "management accountability at every level."
        ),

        "guideline_reference": (
            "Industrial Workforce Safety & PPE Compliance Standard"
        ),
    },
}

# =========================================================
# FALLBACK
# =========================================================

DEFAULT_GUIDELINE = {

    "observation": (
        "A general construction quality deviation was identified during "
        "inspection analysis."
    ),

    "root_cause": {

        "critical": (
            "Root cause analysis indicates a critical underlying deficiency "
            "in construction quality control or engineering compliance that "
            "requires detailed investigation to confirm the precise "
            "contributing factors."
        ),

        "high": (
            "A significant underlying quality or process deficiency is the "
            "probable root cause and should be confirmed through detailed "
            "engineering investigation."
        ),

        "medium": (
            "A moderate underlying quality or procedural deficiency is the "
            "likely root cause and should be evaluated during detailed "
            "inspection."
        ),

        "low": (
            "A minor underlying deficiency, likely procedural or "
            "environmental in nature, is the probable root cause and should "
            "be confirmed during routine inspection."
        ),
    },

    "risk": {

        "critical": (
            "This critical deviation presents an immediate risk to operational "
            "safety and infrastructure integrity requiring urgent intervention."
        ),

        "high": (
            "This high-severity deviation requires prompt corrective action to "
            "prevent further operational or structural impact."
        ),

        "medium": (
            "This moderate deviation may affect operational reliability and "
            "compliance standards if not addressed within a reasonable timeframe."
        ),

        "low": (
            "This low-risk observation presents limited immediate impact but "
            "should be monitored and addressed during routine maintenance."
        ),
    },

    "corrective_action": {

        "critical": (
            "Immediately restrict operations in the affected zone and commission "
            "emergency engineering review. Do not resume normal operations until "
            "written engineering clearance is obtained."
        ),

        "high": (
            "Initiate accelerated corrective action within 48 hours. Engage "
            "qualified engineering personnel to assess and remediate the "
            "identified condition."
        ),

        "medium": (
            "Schedule corrective inspection and repair within 30 days. Apply "
            "appropriate engineering procedures based on detailed assessment "
            "findings."
        ),

        "low": (
            "Address the identified condition during the next scheduled "
            "maintenance cycle. Document observations for ongoing monitoring."
        ),
    },

    "preventive_action": {

        "critical": (
            "Implement continuous monitoring and enhanced inspection protocols "
            "in the affected zone. Review and strengthen quality management "
            "procedures to prevent recurrence."
        ),

        "high": (
            "Establish a structured inspection programme with defined corrective "
            "action thresholds. Review operational procedures to minimise "
            "recurrence risk."
        ),

        "medium": (
            "Continue preventive monitoring and include the identified condition "
            "in the scope of routine inspection procedures."
        ),

        "low": (
            "Maintain standard inspection procedures and monitor the identified "
            "condition during regular inspection cycles."
        ),
    },

    "best_practice": (
        "Maintain enterprise-level construction quality assurance procedures, "
        "conduct regular compliance inspections, and enforce corrective action "
        "accountability at all management levels."
    ),

    "guideline_reference": (
        "InfraGuard Enterprise Quality Assurance Framework"
    ),
}


# =========================================================
# RESOLUTION HELPER
# =========================================================

def _resolve_field(value, severity_key):
    """
    If a field is a severity-stratified dict, return the
    tier matching severity_key.  Falls back to 'medium'
    then to the first available tier, then to empty string.
    Flat string fields are returned as-is.
    """
    if not isinstance(value, dict):
        return value

    return (
        value.get(severity_key)
        or value.get("medium")
        or next(iter(value.values()), "")
    )


def _resolve_confidence_note(confidence):
    """
    Translate an AI detection confidence score into a
    human-readable note for reporting.

    A confidence of exactly 0.0 is treated as "no confidence
    score was supplied" rather than "confidence was measured
    as low" -- callers that don't pass a confidence value
    should not be told their detection is low-confidence.
    """
    if confidence <= 0.0:
        return (
            "No AI confidence score was provided for this finding. "
            "Manual verification recommended."
        )

    if confidence < 0.55:
        return (
            "AI confidence is moderate. "
            "Manual verification recommended."
        )

    if confidence < 0.75:
        return (
            "AI confidence is good. "
            "Visual verification advised."
        )

    return "AI confidence is high."


def recurrence_classification(count):
    """
    Classify how many times an issue type has recurred (e.g. across
    a site's inspection history) into a human-readable pattern label.

    Parameters
    ----------
    count : int
        Number of prior/total occurrences of the issue.

    Returns
    -------
    str
        One of:
        "Critical Recurring Pattern"   (count >= 10)
        "Systemic Recurring Pattern"   (5 <= count < 10)
        "Emerging Recurring Pattern"   (2 <= count < 5)
        "Isolated Occurrence"          (count < 2)
    """
    if count >= 10:
        return "Critical Recurring Pattern"

    if count >= 5:
        return "Systemic Recurring Pattern"

    if count >= 2:
        return "Emerging Recurring Pattern"

    return "Isolated Occurrence"


# =========================================================
# PUBLIC ACCESS
# =========================================================

def get_guidelines(
    issue_type,
    severity="medium",
    confidence=0.0,
):
    """
    Return a flat guideline dict resolved for the given
    issue type, severity tier, and confidence level.

    Parameters
    ----------
    issue_type : str
        Detection label (e.g. 'surface_crack').
    severity : str
        One of 'critical' | 'high' | 'medium' | 'low'.
    confidence : float
        AI detection confidence in [0, 1].  Reserved for
        future confidence-driven text modulation.

    Returns
    -------
    dict with keys:
        observation, root_cause, risk, corrective_action,
        preventive_action, best_practice,
        guideline_reference, potential_consequences,
        management_impact, risk_category, department,
        target_closure_date, audit_priority, audit_exposure,
        executive_recommendation, executive_owner, confidence_note
    """

    template    = GUIDELINES.get(issue_type, DEFAULT_GUIDELINE)
    severity_key = severity.lower()

    if severity_key not in ("critical", "high", "medium", "low"):
        severity_key = "medium"

    return {
        "observation":       _resolve_field(template.get("observation",       DEFAULT_GUIDELINE["observation"]),       severity_key),
        "root_cause":        _resolve_field(template.get("root_cause",        DEFAULT_GUIDELINE["root_cause"]),        severity_key),
        "risk":              _resolve_field(template.get("risk",              DEFAULT_GUIDELINE["risk"]),              severity_key),
        "corrective_action": _resolve_field(template.get("corrective_action", DEFAULT_GUIDELINE["corrective_action"]), severity_key),
        "preventive_action": _resolve_field(template.get("preventive_action", DEFAULT_GUIDELINE["preventive_action"]), severity_key),
        "best_practice":     _resolve_field(template.get("best_practice",     DEFAULT_GUIDELINE["best_practice"]),     severity_key),
        "guideline_reference": template.get(
            "guideline_reference",
            DEFAULT_GUIDELINE["guideline_reference"],
        ),

        # ---- Enterprise Intelligence Layer additions ----
        "potential_consequences": CONSEQUENCE_LIBRARY.get(issue_type, []),

        "management_impact": MANAGEMENT_IMPACT_LIBRARY.get(
            issue_type,
            DEFAULT_MANAGEMENT_IMPACT,
        ),

        "risk_category": RISK_CATEGORY_MAPPING.get(
            severity_key,
            "C",
        ),

        # ---- Upgrade additions (department / timeline / audit / confidence) ----
        "department": DEPARTMENT_MAPPING.get(
            issue_type,
            "Site Management",
        ),

        "target_closure_date": CLOSURE_TIMELINE.get(
            severity_key,
            "30 Days",
        ),

        "audit_priority": AUDIT_PRIORITY.get(
            severity_key,
            "Routine Monitoring",
        ),

        "audit_exposure": AUDIT_EXPOSURE_LIBRARY.get(
            severity_key,
            AUDIT_EXPOSURE_LIBRARY["medium"],
        ),

        "executive_recommendation": EXECUTIVE_RECOMMENDATIONS.get(
            issue_type,
            DEFAULT_EXECUTIVE_RECOMMENDATION,
        ),

        "executive_owner": DEPARTMENT_EXECUTIVE_OWNER.get(
            DEPARTMENT_MAPPING.get(issue_type),
            "Site Manager",
        ),

        "confidence_note": _resolve_confidence_note(confidence),

        
    }