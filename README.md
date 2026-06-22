# InfraGuard – AI Industrial Safety Monitoring System

## Overview
InfraGuard is an AI-powered computer vision system designed to monitor industrial environments and detect safety risks in real time.  
It analyzes video streams using deep learning models to identify PPE compliance violations, dangerous worker-machine proximity, and potential infrastructure hazards.

The system is designed to assist safety teams in construction sites, factories, and industrial environments.

---

## Key Features

- Real-time PPE detection (Helmet, Vest, Goggles, Gloves, Boots)
- Worker detection and machine proximity monitoring
- Heavy machinery detection (Excavator, Bulldozer, Trucks)
- Crack detection for infrastructure monitoring
- Rule-based risk evaluation engine
- Real-time visual alerts
- Multi-camera monitoring support
- Event logging system
- Web dashboard for live monitoring

---

## System Architecture

InfraGuard processes video streams through an AI pipeline that detects safety risks and generates alerts.

python -m src.inference.run_video_inference