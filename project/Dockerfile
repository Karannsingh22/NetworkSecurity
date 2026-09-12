# ---------------------------------------------------------------------------
# Network Security Threat Detection - local Docker image
#
# This image is 100% local: it only installs the Python dependencies in
# requirements.txt and copies the project (including the already-trained
# final_model/ artifacts) into the container, then serves the Streamlit
# app. It does NOT install the AWS CLI, does NOT read AWS environment
# variables, and never talks to S3 / EC2 / ECR or any other cloud service.
# ---------------------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Used only for the container HEALTHCHECK below (curl), not for any cloud access.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer
# whenever only the application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, networksecurity/, Network_Data/,
# data_schema/, final_model/, etc.) - see .dockerignore for what is excluded.
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
