# Dockerfile
# Use Python 3.10 slim image for smaller size
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# System deps some TensorFlow wheels need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv resolves + installs far faster than pip, especially for a package as
# large as tensorflow. Pulled from PyPI's uv wheel, not curl'd from astral.sh.
RUN pip install --no-cache-dir uv
 
# Install Python deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the entire project
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



