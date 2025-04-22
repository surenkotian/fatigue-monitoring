FROM python:3.9-slim

WORKDIR /app

COPY dashboard.py ./dashboard.py
COPY requirements.txt ./requirements.txt
COPY wfh_fatigue_data.csv ./wfh_fatigue_data.csv

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install XlsxWriter

EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
