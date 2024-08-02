# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will use
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]