# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

WORKDIR /app
COPY /app .

EXPOSE 8501

# Install pip requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


CMD [ "streamlit","run","home.py","--server.port=8501", "--server.address=0.0.0.0" ]