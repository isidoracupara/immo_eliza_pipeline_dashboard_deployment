# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

WORKDIR /app

COPY /app .

# Expose default streamlit port 8501
EXPOSE 8501

# Install pip requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit","run"]

CMD ["home.py"]