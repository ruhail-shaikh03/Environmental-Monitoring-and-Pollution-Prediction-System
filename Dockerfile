# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY model/ model/

# Install system dependencies for building some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]
