# Base image
# FROM python:3.12-slim

# # Set working directory
# WORKDIR /app

# # Copy requirements first (better caching)
# #COPY requirements.txt .

# # Install dependencies
# #RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# # Optional: show structure (debug)
# RUN ls -R

# # Default command (example: run inference)
# CMD ["python", "monitoring/monitor.py"]
# #######
# FROM python:3.12-slim

# WORKDIR /app

# # COPY requirements.txt .

# # RUN pip install -r requirements.txt

# COPY . .

# CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
# #######
###

FROM python:3.12-slim

WORKDIR /app_mlops

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app_mlops

CMD ["uvicorn", "model_app.model_inference:app", "--host", "0.0.0.0", "--port", "8000"]