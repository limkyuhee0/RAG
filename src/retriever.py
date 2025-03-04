import json
import os
from dotenv import load_dotenv  # .env 파일 로드
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class VectorDB:
    def __init__(self, data_file="data/processed_texts.json", db_path="data/vector_db"):
        """
        벡터 DB를 구축하고 저장하는 클래스

        Args:
        - data_file (str): JSON 파일에서 문서 리스트 로드
        - db_path (str): 벡터 DB 저장 경로
        """
        self.data_file = data_file
        self.db_path = db_path

        # 🔹 OpenAI API 키 로드 (.env에서 가져옴)
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        # 🔹 OpenAI 임베딩 모델 설정
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.api_key
        )

    def load_documents(self):
        """
        JSON 파일에서 문서 리스트를 로드
        """
        with open(self.data_file, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"📂 {len(documents)}개의 문서 로드 완료")
        return documents

    def create_vector_db(self):
        """
        벡터 DB를 구축하고 저장
        """
        documents = self.load_documents()

        # 🔹 FAISS 벡터 DB 구축
        vectorstore = FAISS.from_texts(documents, self.embedding_model)

        # 🔹 저장 (재사용 가능)
        vectorstore.save_local(self.db_path)
        print(f"✅ FAISS Vector DB 저장 완료: {self.db_path}")

if __name__ == "__main__":
    db = VectorDB()
    db.create_vector_db()
