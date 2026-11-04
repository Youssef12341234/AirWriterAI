FROM python:3.12-slim

RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --chown=user:user requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY --chown=user:user . .

USER user

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]