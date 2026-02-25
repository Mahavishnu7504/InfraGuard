# InfraGuard – AI Industrial Safety Monitoring System

## Overview
InfraGuard is a computer-vision based system for detecting PPE compliance
and safety risks in industrial environments.

## Features
- PPE detection (Helmet, Vest)
- Rule-based risk engine
- Config-driven architecture
- Docker-ready deployment
- CI-tested codebase

## Architecture
[Data] → [YOLO Model] → [Inference] → [Risk Engine] → [Alerts]

## Installation
pip install -r requirements.txt

## Run
python main.py

## Deployment
docker build -t infraguard .
docker run infraguard