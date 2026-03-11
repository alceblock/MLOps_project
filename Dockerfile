# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
#COPY requirements.txt .

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Optional: show structure (debug)
RUN ls -R

# Default command (example: run inference)
CMD ["python", "monitoring/monitor.py"]