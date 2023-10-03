# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in Docker
WORKDIR /app

# Copy the poetry lock and pyproject files
COPY chat_app/pyproject.toml chat_app/poetry.lock ./

# Install poetry
RUN pip install poetry

# Use poetry to install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the application
COPY chat_app .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Command to run the application using gunicorn
#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
