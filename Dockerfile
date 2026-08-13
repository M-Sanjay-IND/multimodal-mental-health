FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    KMP_DUPLICATE_LIB_OK=TRUE

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install python packages
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application source code and artifacts
COPY --chown=user schemas/ $HOME/app/schemas/
COPY --chown=user models/ $HOME/app/models/
COPY --chown=user training/ $HOME/app/training/
COPY --chown=user xai/ $HOME/app/xai/
COPY --chown=user server/ $HOME/app/server/
COPY --chown=user artifacts/ $HOME/app/artifacts/

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Start Uvicorn gateway server on port 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
