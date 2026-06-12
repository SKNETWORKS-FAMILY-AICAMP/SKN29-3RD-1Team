```
# conda 환경 세팅
conda create -f environment.yml
conda activate skn3rd

# .env파일 생성
OPENAI_API_KEY=

# 서버 실행
uvicorn app.main:app --reload

# 이후 localhost:8000/docs 들어가서 확인
```