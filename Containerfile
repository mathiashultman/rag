FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV HF_HOME=/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ladda ner embedding-modellen under byggandet
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', device='cpu')"

COPY . .

RUN mkdir -p /app/chroma

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]