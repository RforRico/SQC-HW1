# 安裝必要的套件
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
