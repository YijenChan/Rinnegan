# OfficeFog Attack-related v1.2

This package contains attack-related observations only for OfficeFog.

Time range: 2026-01-20 09:00:00 to 2026-01-20 15:00:00
Window size: 10 minutes

Files:
- audit_attack_events.csv: attack-related audit events for provenance graph construction.
- network_attack_flows.csv: flow-level network observations aligned with attack windows.
- host_attack_metrics.csv: fine-grained host metric samples around attack activity.
- host_metric_attack_events.csv: extracted metric-clue events for LLM remediation context.
- attack_window_labels.csv: attack-window and corruption-target labels.
- removed_events_plan.csv: planned audit events to remove when constructing the corrupted version.
- attack_event_labels.csv: event-level attack/removal labels.
- attack_entity_labels.csv: malicious canonical entities.
- host_inventory.csv: anonymized host inventory.
- attack_story.json: structured APT story.
- relation_schema.json: provenance relation schema.

Important:
network_attack_flows.csv and host_attack_metrics.csv are not pure window-level statistics.
They preserve fine-grained observations that can be aggregated for model features or directly summarized as LLM window context.
