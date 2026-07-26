FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Install system dependencies needed for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Change the working directory to the user's home
WORKDIR $HOME/app

# Copy the application files with proper ownership
COPY --chown=user . $HOME/app

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Start the Flask application using Gunicorn for better performance
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "flask_app:app"]
