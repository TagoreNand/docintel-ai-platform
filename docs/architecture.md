# Architecture Overview

## System goals

- Process enterprise documents end to end
- Extract structured intelligence
- Support grounded search and QA
- Detect anomalies
- Route uncertain cases to human review
- Remain local-dev friendly while preserving production-grade structure

## Logical components

### 1. API Gateway
Receives uploads, search requests, review actions, and analytics queries.

### 2. Ingestion Pipeline
Parses the file, normalizes text, classifies the document, extracts entities, computes anomaly scores, and indexes chunks.

### 3. Persistence Layer
Stores:
- document metadata
- extracted entities
- text chunks
- review tasks

### 4. Retrieval Layer
Runs lexical + vector-style scoring over document chunks and returns evidence-backed results.

### 5. Review Layer
Creates tasks when confidence or anomaly thresholds are breached.

### 6. Frontend
Operational console for upload, search, and review workflows.

## Service boundaries in a production deployment

- ingestion-service
- extraction-service
- retrieval-service
- anomaly-service
- review-service
- analytics-service

This repository keeps them as modules in one backend for easy local execution while preserving a realistic architecture.
