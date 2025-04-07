# Segmentation Application with U-Net

## Overview
This project is a complete segmentation application that includes both a frontend and backend powered by a trained U-Net model. The backend performs image segmentation using a PyTorch-based U-Net model, while the frontend provides a user interface to upload images and view results.

## Project Structure
segmentation-app/ │-- backend/ # FastAPI backend using U-Net model │-- frontend/ # React (or JS/CSS/HTML) frontend │-- docker-compose.yml # Docker Compose configuration │-- README.md # Documentation


## Features
- Upload images from a web interface
- Backend serves segmentation using a trained U-Net model
- Real-time visualization of segmented results
- Packaged using Docker and easily deployable

---

## Getting Started with Docker

### Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/)
- Install [Docker Compose](https://docs.docker.com/compose/install/)

### Build and Run the Application
```bash
docker-compose up -d --build
