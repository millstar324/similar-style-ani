# 베이스 이미지 설정 (python 3.9 버전 사용 예시)
FROM python:3.8.19

# 작업 디렉토리 생성
WORKDIR /app

# 필요하면 시스템 패키지 설치 (옵션)
# RUN apt-get update && apt-get install -y ...

# 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

# 스트림릿 포트 노출 (기본 8501)
EXPOSE 8501

# 컨테이너 실행 시 커맨드
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
