from collections import Counter


_ISSUE_DISPLAY_NAMES = {
    "surface_crack":      "Surface Cracking",
    "corrosion":          "Corrosion",
    "water_leakage":      "Water Leakage",
    "rebar_exposure":     "Rebar Exposure",
    "material_damage":    "Material Damage",
    "poor_housekeeping":  "Housekeeping Deficiencies",
    "ppe_non_compliance": "PPE Non-Compliance",
}

_SEVERITY_ORDER = {
    "critical": 0,
    "high":     1,
    "medium":   2,
    "low":      3,
}

# Issue category classification for systemic analysis
_STRUCTURAL_ISSUES = {"surface_crack", "corrosion", "water_leakage", "rebar_exposure", "material_damage"}
_SAFETY_ISSUES     = {"poor_housekeeping", "ppe_non_compliance"}

# Audit exposure thresholds
_HIGH_AUDIT_RISK_ISSUES = {"ppe_non_compliance", "poor_housekeeping", "rebar_exposure"}


def _display_name(issue_type: str) -> str:
    return _ISSUE_DISPLAY_NAMES.get(
        issue_type,
        issue_type.replace("_", " ").title(),
    )


# =========================================================
# INSPECTION PROFILE BUILDER
# =========================================================

def _build_profile(findings: list, compliance_score: int) -> dict:
    """
    Derives a structured inspection profile from the raw
    finding list.  All summary sections draw from this
    profile rather than from the findings list directly,
    keeping each section generator simple and testable.

    Enhancement 6: adds most_common_issue, recurring_issues,
    and confidence_reliability to the profile.
    """

    severity_counts = Counter()
    issue_severity  = {}       # issue_type → best severity rank
    issue_freq      = Counter()
    confidence_sum  = 0.0
    corrective_set  = {}       # issue_type → corrective_action (highest sev)

    for item in findings:

        sev        = item.get("severity",   "medium").lower()
        issue_type = item.get("issue_type", "unknown")
        confidence = float(item.get("confidence", 0.75))
        corrective = item.get("corrective_action", "")

        if sev not in _SEVERITY_ORDER:
            sev = "medium"

        severity_counts[sev] += 1
        issue_freq[issue_type] += 1
        confidence_sum += confidence

        # Keep the worst severity seen for each issue type
        current_rank = issue_severity.get(issue_type, 99)
        new_rank     = _SEVERITY_ORDER[sev]
        if new_rank < current_rank:
            issue_severity[issue_type]  = new_rank
            corrective_set[issue_type]  = corrective

    total = len(findings)
    avg_confidence = round(confidence_sum / total, 2) if total > 0 else 0.0

    # Rank issues: worst severity first, then highest frequency
    ranked_issues = sorted(
        issue_severity.keys(),
        key=lambda x: (issue_severity[x], -issue_freq[x]),
    )

    # Determine dominant severity
    if severity_counts.get("critical", 0) > 0:
        dominant_severity = "critical"
    elif severity_counts.get("high", 0) > 0:
        dominant_severity = "high"
    elif severity_counts.get("medium", 0) > 0:
        dominant_severity = "medium"
    else:
        dominant_severity = "low"

    # Compliance class
    if compliance_score >= 90:
        compliance_class = "excellent"
    elif compliance_score >= 75:
        compliance_class = "moderate"
    elif compliance_score >= 50:
        compliance_class = "poor"
    else:
        compliance_class = "critical"

    # ── Enhancement 6: Pattern Analytics ─────────────────
    # most_common_issue: issue type with highest occurrence count
    most_common_issue = (
        issue_freq.most_common(1)[0][0]
        if issue_freq else None
    )

    # recurring_issues: issue types appearing more than once,
    # sorted by frequency descending
    recurring_issues = {
        it: cnt
        for it, cnt in issue_freq.items()
        if cnt > 1
    }
    recurring_issues = dict(
        sorted(recurring_issues.items(), key=lambda x: -x[1])
    )

    # ── Enhancement 7: Confidence Reliability ─────────────
    if avg_confidence >= 0.90:
        confidence_reliability = "Enterprise Verified"
    elif avg_confidence >= 0.80:
        confidence_reliability = "High Confidence"
    elif avg_confidence >= 0.70:
        confidence_reliability = "Moderate Confidence"
    else:
        confidence_reliability = "Review Recommended"

    return {
        "total":                  total,
        "compliance_score":       compliance_score,
        "compliance_class":       compliance_class,
        "dominant_severity":      dominant_severity,
        "severity_counts":        severity_counts,
        "ranked_issues":          ranked_issues,
        "issue_freq":             issue_freq,
        "issue_severity":         issue_severity,
        "avg_confidence":         avg_confidence,
        "confidence_reliability": confidence_reliability,
        "corrective_set":         corrective_set,
        "most_common_issue":      most_common_issue,
        "recurring_issues":       recurring_issues,
    }


# =========================================================
# SECTION BUILDERS  (original — untouched)
# =========================================================

def _opening_paragraph(profile: dict) -> str:
    """
    Context-aware opening that sets the tone for the
    inspection result.
    """

    score = profile["compliance_score"]
    dom   = profile["dominant_severity"]
    cc    = profile["compliance_class"]

    if cc == "excellent":
        return (
            f"InfraGuard Enterprise AI completed construction quality inspection "
            f"analysis and generated an overall compliance score of {score}/100. "
            f"The inspected environment demonstrates strong operational discipline "
            f"and acceptable infrastructure compliance conditions."
        )

    if cc == "moderate":
        return (
            f"InfraGuard Enterprise AI completed construction quality inspection "
            f"analysis and generated an overall compliance score of {score}/100. "
            f"Inspection analytics identified operational deviations requiring "
            f"structured corrective and preventive action."
        )

    if cc == "poor":
        return (
            f"InfraGuard Enterprise AI completed construction quality inspection "
            f"analysis and generated an overall compliance score of {score}/100. "
            f"The inspection identified significant infrastructure compliance "
            f"concerns that require prompt corrective intervention."
        )

    # critical
    return (
        f"InfraGuard Enterprise AI completed construction quality inspection "
        f"analysis and generated an overall compliance score of {score}/100. "
        f"Critical-risk operational deviations and construction quality deficiencies "
        f"were identified, requiring immediate engineering and management intervention."
    )


def _findings_paragraph(profile: dict) -> str:
    """
    Names the top issues found, ranked by severity, and
    provides a severity distribution summary.
    """

    ranked  = profile["ranked_issues"]
    freq    = profile["issue_freq"]
    sc      = profile["severity_counts"]
    total   = profile["total"]

    # Build issue name list (up to 4)
    top_issues = [_display_name(i) for i in ranked[:4]]

    if len(top_issues) == 1:
        issue_str = top_issues[0]
    elif len(top_issues) == 2:
        issue_str = f"{top_issues[0]} and {top_issues[1]}"
    else:
        issue_str = (
            ", ".join(top_issues[:-1])
            + f", and {top_issues[-1]}"
        )

    # Severity distribution line
    sev_parts = []
    if sc.get("critical", 0) > 0:
        n = sc["critical"]
        sev_parts.append(f"{n} critical")
    if sc.get("high", 0) > 0:
        n = sc["high"]
        sev_parts.append(f"{n} high-priority")
    if sc.get("medium", 0) > 0:
        n = sc["medium"]
        sev_parts.append(f"{n} moderate")
    if sc.get("low", 0) > 0:
        n = sc["low"]
        sev_parts.append(f"{n} low-risk")

    if len(sev_parts) == 1:
        sev_str = sev_parts[0]
    elif len(sev_parts) == 2:
        sev_str = f"{sev_parts[0]} and {sev_parts[1]}"
    else:
        sev_str = ", ".join(sev_parts[:-1]) + f", and {sev_parts[-1]}"

    para = (
        f"A total of {total} quality deviation{'s were' if total != 1 else ' was'} "
        f"identified across the following categories: {issue_str}. "
        f"The severity distribution comprises {sev_str} "
        f"finding{'s' if total != 1 else ''}."
    )

    # Enhancement 7: confidence_reliability inline
    conf = profile["avg_confidence"]
    cr   = profile["confidence_reliability"]
    if conf >= 0.90:
        para += (
            f" Average AI detection confidence is {int(conf * 100)}% "
            f"({cr}), indicating strong visual evidence reliability "
            f"across the identified findings."
        )
    elif conf < 0.70:
        para += (
            f" Average AI detection confidence is {int(conf * 100)}% "
            f"({cr}). Field verification by a qualified inspector is "
            f"recommended for findings with lower confidence scores."
        )

    return para


def _risk_paragraph(profile: dict) -> str:
    """
    Produces a risk statement scaled to the dominant severity
    and number of findings at the top severity level.

    Enhancement 8: adds immediate risk, future risk, and
    compliance risk language.
    """

    dom = profile["dominant_severity"]
    sc  = profile["severity_counts"]

    if dom == "critical":
        n = sc.get("critical", 0)
        return (
            f"{n} critical-risk condition{'s' if n != 1 else ''} "
            f"{'were' if n != 1 else 'was'} identified that may significantly affect "
            f"operational safety, infrastructure reliability, and structural integrity "
            f"if not resolved immediately. Continued operation without corrective "
            f"intervention presents elevated safety and liability exposure. "
            f"Immediate risk to site personnel and structural systems is confirmed. "
            f"Future risk includes accelerated infrastructure degradation and "
            f"permanent structural compromise if remediation is deferred. "
            f"Compliance risk is severe — unresolved critical findings may result "
            f"in mandatory regulatory intervention, operational shutdown orders, "
            f"and adverse external audit outcomes."
        )

    if dom == "high":
        n = sc.get("high", 0)
        return (
            f"{n} high-priority deviation{'s require' if n != 1 else ' requires'} "
            f"accelerated corrective response and preventive engineering supervision. "
            f"Operational risk exposure remains elevated and may escalate if "
            f"corrective action is deferred beyond the current operational cycle. "
            f"Future risk includes progressive infrastructure deterioration and "
            f"increased likelihood of critical escalation during subsequent inspections. "
            f"Compliance risk is elevated — current conditions may attract "
            f"adverse observations during external HSE and quality audits."
        )

    if dom == "medium":
        n = sc.get("medium", 0)
        return (
            f"{n} moderate infrastructure deviation{'s were' if n != 1 else ' was'} "
            f"identified and should be addressed through structured preventive "
            f"quality workflows. Risk exposure is manageable under a defined "
            f"corrective maintenance programme. "
            f"If left unresolved, recurring deficiencies may result in accelerated "
            f"infrastructure degradation and increased compliance risk over successive "
            f"inspection cycles."
        )

    n = sc.get("low", 0)
    return (
        f"{n} low-risk observation{'s were' if n != 1 else ' was'} identified "
        f"with limited immediate operational impact. Current inspection conditions "
        f"indicate controlled operational quality with minimal infrastructure "
        f"concerns."
    )


def _analytics_paragraph(profile: dict) -> str:
    """
    Structured analytics block showing the full severity
    distribution clearly.
    """

    sc    = profile["severity_counts"]
    total = profile["total"]
    conf  = profile["avg_confidence"]
    cr    = profile["confidence_reliability"]

    lines = [
        "Inspection analytics identified:",
        f"  • Critical Findings:        {sc.get('critical', 0)}",
        f"  • High Severity Findings:   {sc.get('high',     0)}",
        f"  • Medium Findings:          {sc.get('medium',   0)}",
        f"  • Low Findings:             {sc.get('low',      0)}",
        f"  • Total Findings:           {total}",
        f"  • Average AI Confidence:    {int(conf * 100)}%",
        f"  • Confidence Reliability:   {cr}",
    ]

    return "\n".join(lines)


def _recommendation_paragraph(profile: dict) -> str:
    """
    Produces a recommendation statement that is specific to
    the dominant severity and includes corrective timelines.
    """

    dom      = profile["dominant_severity"]
    ranked   = profile["ranked_issues"]
    corr_set = profile["corrective_set"]

    # Primary corrective action from the highest-severity issue
    primary_issue      = ranked[0] if ranked else None
    primary_corrective = corr_set.get(primary_issue, "") if primary_issue else ""

    if dom == "critical":
        base = (
            "Immediate engineering review, operational restriction assessment, "
            "and executive corrective action enforcement are required. "
            "Operations must not resume in affected zones until formal written "
            "engineering clearance is obtained."
        )
        if primary_corrective:
            base += f" Priority corrective guidance: {primary_corrective}"
        return base

    if dom == "high":
        base = (
            "Accelerated corrective action within 14 days and preventive "
            "inspection reinforcement are recommended to prevent further "
            "infrastructure degradation. Engineering supervision should be "
            "engaged for all identified high-priority conditions."
        )
        if primary_corrective:
            base += f" Priority corrective guidance: {primary_corrective}"
        return base

    if dom == "medium":
        return (
            "Preventive maintenance workflows and structured monitoring "
            "procedures are recommended within the next 30 days. "
            "Routine inspection reinforcement and controlled corrective "
            "maintenance activities should be scheduled and tracked."
        )

    return (
        "Continue standard enterprise inspection workflows and routine "
        "compliance monitoring procedures. Maintain periodic quality "
        "assurance verification and operational discipline enforcement."
    )


def _closing_statement(profile: dict) -> str:
    """
    Closing statement that names the applicable standard
    and frames the report's purpose without over-engineering
    clean results.
    """

    cc = profile["compliance_class"]

    if cc in ("critical", "poor"):
        return (
            "This inspection report was generated using InfraGuard Enterprise "
            "AI Inspection Intelligence. The findings and corrective guidance "
            "contained in this report require prompt review by qualified "
            "engineering and site management personnel."
        )

    return (
        "This inspection report was generated using InfraGuard Enterprise "
        "AI Inspection Intelligence for audit-ready construction quality "
        "assurance, operational compliance verification, and infrastructure "
        "risk management workflows."
    )


# =========================================================
# CLEAN INSPECTION SUMMARY
# =========================================================

def _no_findings_summary(compliance_score: int) -> str:
    """
    Returns a concise, appropriately positive summary for
    inspections with no significant findings.  Avoids
    over-engineering the language for a clean result.
    """

    return (
        f"InfraGuard Enterprise AI completed construction quality inspection "
        f"analysis with an overall compliance score of {compliance_score}/100. "
        f"No significant operational defects or structural compliance deviations "
        f"were identified during the inspection.\n\n"
        f"Inspection intelligence indicates stable infrastructure conditions "
        f"with acceptable enterprise-level operational quality alignment. "
        f"Routine preventive monitoring and periodic inspection workflows are "
        f"recommended to maintain long-term infrastructure reliability and "
        f"sustained compliance performance."
    )


# =========================================================
# NEW SECTION BUILDERS  (Enhancements 1–5, 9)
# =========================================================

def _executive_board_paragraph(profile: dict) -> str:
    """
    Enhancement 9: One-paragraph leadership-level summary
    written in board-appropriate language. Concise, direct,
    and governance-focused.
    """

    score  = profile["compliance_score"]
    cc     = profile["compliance_class"]
    dom    = profile["dominant_severity"]
    total  = profile["total"]
    ranked = profile["ranked_issues"]

    top_issue_str = (
        _display_name(ranked[0]) if ranked else "quality deviations"
    )

    if cc == "critical":
        return (
            f"This inspection has identified {total} construction quality "
            f"control deficiencies, including critical-severity conditions, "
            f"that require immediate executive oversight and governance response. "
            f"The compliance score of {score}/100 reflects unacceptable risk "
            f"exposure across the inspected site. {top_issue_str} represents "
            f"the highest-priority concern requiring Board-level corrective "
            f"action accountability and formal engineering clearance before "
            f"operations may resume."
        )

    if cc == "poor":
        return (
            f"The inspection identified {total} quality control deviations "
            f"resulting in a compliance score of {score}/100, indicating "
            f"significant management attention is required. "
            f"Recurring infrastructure and safety compliance gaps — particularly "
            f"{top_issue_str} — represent elevated operational and reputational "
            f"risk that requires structured corrective action governance and "
            f"defined management accountability."
        )

    if cc == "moderate":
        return (
            f"The inspection identified {total} operational quality deviations "
            f"with a compliance score of {score}/100. While conditions are not "
            f"immediately critical, the inspection indicates recurring quality "
            f"control deficiencies that require management oversight and "
            f"structured corrective action governance to prevent escalation "
            f"during subsequent inspection cycles."
        )

    # excellent
    return (
        f"The inspection returned a compliance score of {score}/100, "
        f"reflecting strong operational quality management performance. "
        f"No executive escalation is required at this time. Continued "
        f"commitment to preventive quality governance will sustain "
        f"the current compliance posture."
    )


def _recurring_issue_paragraph(profile: dict) -> str:
    """
    Enhancement 1: Identifies issue types that appear more
    than once and characterises them as systemic weaknesses
    rather than isolated events.
    """

    recurring = profile["recurring_issues"]
    most_common = profile["most_common_issue"]

    if not recurring:
        return ""

    # Build display-name list sorted by frequency
    recurring_display = [
        (f"{_display_name(it)} ({cnt} occurrences)")
        for it, cnt in recurring.items()
    ]

    if len(recurring_display) == 1:
        issues_str = recurring_display[0]
    elif len(recurring_display) == 2:
        issues_str = f"{recurring_display[0]} and {recurring_display[1]}"
    else:
        issues_str = (
            ", ".join(recurring_display[:-1])
            + f", and {recurring_display[-1]}"
        )

    most_common_name = _display_name(most_common) if most_common else ""

    lead = (
        f"Recurring issue analysis identified the following deficiency "
        f"pattern{'s' if len(recurring) > 1 else ''} across the inspection "
        f"dataset: {issues_str}. "
    )

    if most_common_name:
        lead += (
            f"The most frequently observed condition is {most_common_name}, "
            f"indicating a systemic weakness in site quality enforcement "
            f"protocols that extends beyond isolated occurrence. "
        )

    lead += (
        f"Recurring deficiencies of this nature are indicative of "
        f"underlying process failures rather than one-off deviations, "
        f"and require systemic corrective intervention rather than "
        f"item-by-item remediation alone."
    )

    return lead


def _management_attention_paragraph(profile: dict) -> str:
    """
    Enhancement 2: Identifies issues that require named
    management attention, using executive-register language
    aligned to audit exposure and safety culture.
    """

    ranked   = profile["ranked_issues"]
    issue_sv = profile["issue_severity"]
    sc       = profile["severity_counts"]
    cc       = profile["compliance_class"]
    recurring = profile["recurring_issues"]

    if not ranked:
        return ""

    # Issues at critical or high severity warrant explicit management call-out
    mgmt_issues = [
        it for it in ranked
        if issue_sv.get(it, 99) <= 1  # critical (0) or high (1)
    ]

    # Also flag recurring issues regardless of severity
    recurring_mgmt = [
        it for it in recurring
        if it not in mgmt_issues
    ]

    all_mgmt = mgmt_issues + recurring_mgmt

    if not all_mgmt:
        # Nothing rises to management attention in a clean or low-sev result
        if cc in ("excellent", "moderate"):
            return ""
        # Fall back to any ranked issues
        all_mgmt = ranked[:3]

    mgmt_names = [_display_name(it) for it in all_mgmt[:4]]

    if len(mgmt_names) == 1:
        issues_str = mgmt_names[0]
    elif len(mgmt_names) == 2:
        issues_str = f"{mgmt_names[0]} and {mgmt_names[1]}"
    else:
        issues_str = ", ".join(mgmt_names[:-1]) + f", and {mgmt_names[-1]}"

    audit_exposure_issues = [
        _display_name(it) for it in all_mgmt
        if it in _HIGH_AUDIT_RISK_ISSUES
    ]

    base = (
        f"Management attention is required for {issues_str}. "
    )

    if audit_exposure_issues:
        ae_str = (
            audit_exposure_issues[0]
            if len(audit_exposure_issues) == 1
            else ", ".join(audit_exposure_issues[:-1]) + f" and {audit_exposure_issues[-1]}"
        )
        base += (
            f"In particular, {ae_str} "
            f"{'carry' if len(audit_exposure_issues) > 1 else 'carries'} "
            f"elevated audit exposure and safety culture implications "
            f"that extend beyond infrastructure risk into regulatory "
            f"and reputational risk territory. "
        )

    if recurring:
        base += (
            f"The recurring nature of identified deficiencies suggests "
            f"that current supervisory enforcement mechanisms require "
            f"strengthening to prevent continued non-compliance."
        )
    else:
        base += (
            f"Defined management accountability and formal corrective "
            f"action closure tracking are recommended to ensure "
            f"identified conditions are fully remediated within "
            f"committed timelines."
        )

    return base


def _systemic_deficiency_paragraph(profile: dict) -> str:
    """
    Enhancement 3: Moves the narrative from individual findings
    to system-level analysis — identifying which operational
    systems are failing, not just which items are defective.
    """

    ranked   = profile["ranked_issues"]
    issue_sv = profile["issue_severity"]
    recurring = profile["recurring_issues"]
    total    = profile["total"]

    if not ranked:
        return ""

    # Classify all found issues into systems
    structural_found = [it for it in ranked if it in _STRUCTURAL_ISSUES]
    safety_found     = [it for it in ranked if it in _SAFETY_ISSUES]
    other_found      = [
        it for it in ranked
        if it not in _STRUCTURAL_ISSUES and it not in _SAFETY_ISSUES
    ]

    system_lines = []

    if structural_found:
        names = [_display_name(it) for it in structural_found]
        system_lines.append(
            f"  • Structural Quality Control "
            f"({', '.join(names)})"
        )

    if safety_found:
        names = [_display_name(it) for it in safety_found]
        system_lines.append(
            f"  • Site Safety and Compliance Management "
            f"({', '.join(names)})"
        )

    if other_found:
        names = [_display_name(it) for it in other_found]
        system_lines.append(
            f"  • General Quality Assurance "
            f"({', '.join(names)})"
        )

    if not system_lines:
        return ""

    # Determine whether weaknesses are recurring or isolated
    has_recurring = bool(recurring)
    pattern_note = (
        "recurring patterns across multiple findings indicate systemic "
        "process failures within these operational systems"
        if has_recurring else
        "isolated findings across these operational systems indicate "
        "localised quality control gaps"
    )

    intro = (
        f"The inspection indicates {pattern_note}. "
        f"The following operational systems demonstrate quality "
        f"control weaknesses requiring structured corrective governance:\n"
    )

    closing = (
        "\nAddressing these systemic weaknesses requires process-level "
        "intervention — including supervisory reinforcement, procedural "
        "updates, and management accountability frameworks — rather than "
        "item-level corrective action alone."
    )

    return intro + "\n".join(system_lines) + closing


def _audit_exposure_paragraph(profile: dict) -> str:
    """
    Enhancement 4: HSE consultant-register language describing
    the audit risk created by current findings. Driven by
    compliance_score and severity_counts.
    """

    score    = profile["compliance_score"]
    sc       = profile["severity_counts"]
    cc       = profile["compliance_class"]
    ranked   = profile["ranked_issues"]
    recurring = profile["recurring_issues"]

    crit_n = sc.get("critical", 0)
    high_n = sc.get("high",     0)

    # Identify issues with high audit exposure
    audit_risk_present = [
        _display_name(it) for it in ranked
        if it in _HIGH_AUDIT_RISK_ISSUES
    ]

    if cc == "excellent" and crit_n == 0 and high_n == 0:
        return (
            f"The current compliance score of {score}/100 presents "
            f"low audit exposure. The site demonstrates acceptable "
            f"quality alignment and is well-positioned to withstand "
            f"external quality and HSE audit scrutiny."
        )

    base = (
        f"Current findings present a measurable audit exposure risk. "
        f"A compliance score of {score}/100 "
    )

    if cc == "critical":
        base += (
            f"and the presence of {crit_n} critical finding{'s' if crit_n != 1 else ''} "
            f"are likely to result in mandatory adverse observations, "
            f"corrective action notices, and potential operational suspension "
            f"directives during any external quality or regulatory audit."
        )
    elif cc == "poor":
        base += (
            f"and {crit_n + high_n} critical and high-severity finding{'s' if (crit_n + high_n) != 1 else ''} "
            f"may attract significant adverse observations during external "
            f"quality, HSE, and regulatory compliance audits."
        )
    else:
        base += (
            f"reflects conditions that may result in qualified observations "
            f"during external quality and HSE audit reviews if corrective "
            f"action is not completed prior to audit."
        )

    if audit_risk_present:
        ar_str = (
            audit_risk_present[0]
            if len(audit_risk_present) == 1
            else ", ".join(audit_risk_present[:-1]) + f" and {audit_risk_present[-1]}"
        )
        base += (
            f" {ar_str} "
            f"{'carry' if len(audit_risk_present) > 1 else 'carries'} "
            f"particularly high audit scrutiny risk due to "
            f"regulatory prominence in HSE and construction quality frameworks."
        )

    if recurring:
        base += (
            f" The presence of recurring deficiencies across multiple "
            f"findings is likely to be interpreted by external auditors "
            f"as evidence of systemic management control failure rather "
            f"than isolated quality deviations."
        )

    return base


def _strategic_recommendation_paragraph(profile: dict) -> str:
    """
    Enhancement 5: Management-improvement-oriented strategic
    recommendations, complementing the corrective-action focus
    of _recommendation_paragraph().
    """

    dom      = profile["dominant_severity"]
    cc       = profile["compliance_class"]
    ranked   = profile["ranked_issues"]
    recurring = profile["recurring_issues"]
    sc       = profile["severity_counts"]

    # Build a context-appropriate set of strategic actions
    actions = []

    # Always recommend these foundational governance items
    actions.append("  • Formal corrective action register with assigned ownership and closure deadlines")
    actions.append("  • Weekly site quality inspection and supervisor sign-off protocol")

    # Safety-specific
    safety_found = [it for it in ranked if it in _SAFETY_ISSUES]
    if safety_found:
        if "ppe_non_compliance" in ranked:
            actions.append("  • PPE compliance monitoring programme with daily toolbox talks")
        if "poor_housekeeping" in ranked:
            actions.append("  • Weekly housekeeping audit with photographic evidence records")

    # Structural-specific
    struct_found = [it for it in ranked if it in _STRUCTURAL_ISSUES]
    if struct_found:
        actions.append("  • Qualified engineering review and periodic structural condition assessment")
        if "surface_crack" in ranked or "rebar_exposure" in ranked:
            actions.append("  • Concrete condition monitoring programme and cover depth verification")

    # Recurring-specific
    if recurring:
        actions.append("  • Root cause analysis for all recurring deficiency categories")
        actions.append("  • Supervisory enforcement reinforcement and accountability escalation protocol")

    # Severity-specific escalation
    if dom in ("critical", "high"):
        actions.append("  • Management review meeting within 7 days to assess corrective action status")
        actions.append("  • Pre-audit internal quality review prior to any scheduled external inspection")

    action_block = "\n".join(actions)

    if cc in ("critical", "poor"):
        intro = (
            "Recommended strategic management actions to address "
            "identified systemic quality control weaknesses and reduce "
            "ongoing audit and operational risk:"
        )
    else:
        intro = (
            "Recommended strategic actions to sustain and improve "
            "quality management performance:"
        )

    return f"{intro}\n{action_block}"


# =========================================================
# PUBLIC SUMMARY ENGINE
# =========================================================

def generate_llm_summary(
    findings:         list,
    compliance_score: int,
) -> str:
    """
    Generate a deterministic, findings-driven enterprise
    inspection intelligence summary for a single-image
    inspection result.

    Parameters
    ----------
    findings : list
        Processed finding dicts from generate_report().
    compliance_score : int
        Compliance score in [0, 100].

    Returns
    -------
    str — multi-paragraph plain-text summary ready for
          display in the analysis panel and PDF report.
    """

    if not findings:
        return _no_findings_summary(compliance_score)

    profile = _build_profile(findings, compliance_score)

    # Build all sections; empty strings are filtered out
    opening               = _opening_paragraph(profile)
    executive_board       = _executive_board_paragraph(profile)
    findings_para         = _findings_paragraph(profile)
    recurring_issue       = _recurring_issue_paragraph(profile)
    risk                  = _risk_paragraph(profile)
    management_attention  = _management_attention_paragraph(profile)
    systemic_deficiencies = _systemic_deficiency_paragraph(profile)
    audit_exposure        = _audit_exposure_paragraph(profile)
    analytics             = _analytics_paragraph(profile)
    recommendation        = _recommendation_paragraph(profile)
    strategic_recs        = _strategic_recommendation_paragraph(profile)
    closing               = _closing_statement(profile)

    sections = [
        opening,
        executive_board,
        findings_para,
        recurring_issue,
        risk,
        management_attention,
        systemic_deficiencies,
        audit_exposure,
        analytics,
        recommendation,
        strategic_recs,
        closing,
    ]

    return "\n\n".join(s for s in sections if s.strip())