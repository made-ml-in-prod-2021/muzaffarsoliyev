FROM python:3.6
COPY requirements.txt ./requirements.txt
COPY setup.py ./setup.py
RUN pip install -r requirements.txt

COPY configs/ /configs/
COPY api/ /api/
COPY models/ /models/
COPY src/ /src/

WORKDIR .

ENV PATH_TO_MODEL="models/model.pkl"
ENV PATH_TO_TRANSFORMER="models/transformer.pkl"
ENV PORT 80

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "80"]