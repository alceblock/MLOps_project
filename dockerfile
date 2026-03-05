# Official Python image from Docker Hub
FROM python:3.12-slim

# Copy the current directory contents (app.py) into the container at path /app
COPY ./CI_CD/CD /app

# Set working directory
WORKDIR /app
RUN ls

# Run Python script
CMD ["python", "app.py"]