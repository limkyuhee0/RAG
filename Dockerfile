# 1. Python 3.12 기반 컨테이너 생성
FROM python:3.12

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 업데이트 및 Poetry 설치
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

# 4. 환경 변수 설정 (Poetry를 PATH에 추가)
ENV PATH="/root/.local/bin:$PATH"

# 5. 필요 없는 파일 제외를 위해 .dockerignore 활용 (추가 필요)
COPY pyproject.toml poetry.lock ./

# 6. Poetry 종속성 설치 (개발 의존성 제외)
RUN poetry install --no-root --no-interaction --no-ansi
# RUN poetry install --only=main

# 7. 나머지 코드 및 실행 스크립트 복사
COPY . .

# 8. 실행 전 output 디렉토리 생성
RUN mkdir -p /app/output

# 9. 실행 권한 부여
RUN chmod +x run.sh

# 10. 실행 명령어 변경 (ENTRYPOINT 사용)
# ENTRYPOINT ["/bin/bash", "./run.sh"]
# CMD ["/bin/bash", "-c", "./run.sh"]
CMD ["./run.sh"]