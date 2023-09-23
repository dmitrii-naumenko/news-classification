FROM python:3.11

COPY requirements.txt setup.py /workdir/
WORKDIR /workdir
RUN pip install -U -e .

COPY app/ /workdir/app/
COPY ml/ /workdir/ml/
COPY model/ /workdir/model/


RUN python -m nltk.downloader 'punkt'

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]