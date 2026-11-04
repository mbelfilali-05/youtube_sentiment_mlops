FROM python:3.10-slim

# HF Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies first: this layer is cached unless requirements change
COPY --chown=user requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Only the files the API needs
COPY --chown=user src/__init__.py src/__init__.py
COPY --chown=user src/api/ src/api/
COPY --chown=user src/utils/ src/utils/
COPY --chown=user app_api.py .
COPY --chown=user models/sentiment_pipeline.joblib models/sentiment_pipeline.joblib

EXPOSE 7860
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]