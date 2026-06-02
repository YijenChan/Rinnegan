# OfficeFog v1.2 with Background

OfficeFog v1.2 is a controlled five-host office-network dataset for testing robust APT detection under corrupted log integrity.

## Scope

- Time range: 2026-01-20 09:00:00 to 2026-01-20 15:00:00
- Window size: 10 minutes
- Number of windows: 36
- Hosts: 5
- Modalities: audit logs, network flows, host metrics, host metric clue events

## Important design choice

This dataset includes both clean and corrupted versions.

- `clean/audit_log.csv` contains benign background audit events and attack audit events.
- `corrupted/audit_log.csv` removes selected attack-related audit events listed in `labels/removed_events.csv`.
- Network flows and host metrics are kept unchanged in the corrupted version, following the threat model that side views are harder to consistently rewrite.

## Background realism

The benign background includes:
- routine finance workstation activity,
- developer SSH/git/docker activity,
- file-server SMB/NFS/backup activity,
- portal nginx/uwsgi activity,
- monitoring and log collection traffic,
- benign high-noise bursts such as docker build, dependency install, backup snapshot, portal cache refresh, and software update.

## Attack overview

The APT begins on `ws-finance-01`, executes `update-check`, escalates privilege through a misconfigured `maintenance-helper`, installs `syswatch`, collects files from `file-srv-01`, touches `portal-web-01` with `portal-sync`, performs C2 with `198.51.100.77:443`, and exfiltrates to `203.0.113.88:443`.

## Scale

- Clean audit events: 5656
- Corrupted audit events: 5641
- Network flows: 1373
- Host metric samples: 1800
- Host metric clue events: 20
- Removed audit events: 15
- Entity labels: 545
