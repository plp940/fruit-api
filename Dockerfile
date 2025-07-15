FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
