import pymupdf  # pymupdf
import json
import os
import re

class PDFProcessor:
    def __init__(self, input_folder="data/raw_pdfs", output_file="data/processed_texts.json"):
        """
        PDF 문서를 처리하는 클래스

        Args:
        - input_folder (str): 원본 PDF 파일이 저장된 폴더
        - output_file (str): 처리된 텍스트 리스트가 저장될 JSON 파일 경로
        """
        self.input_folder = input_folder
        self.output_file = output_file

    def extract_text_from_pdf(self, pdf_path):
        """
        PDF 파일에서 텍스트를 추출하는 메서드
        - pymupdf (fitz)를 사용하여 모든 페이지의 텍스트를 가져옴
        """
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text

    def clean_text(self, text):
        """
        텍스트 정제:
        - `< >`, `[ ]` 괄호 안 내용 제거
        - 여러 개의 공백을 하나의 공백으로 변환
        - 온점(`.`) 기준으로 문장을 리스트로 분할
        """
        text = re.sub(r'<.*?>|\[.*?\]', '', text)  # 괄호 및 내부 내용 제거
        text = re.sub(r'\s+', ' ', text).strip()  # 연속된 공백 제거
        return text.split('.')  # 온점(`.`) 기준으로 리스트 변환

    def process_pdfs(self):
        """
        지정된 폴더 내 모든 PDF 파일을 처리하여 JSON 파일로 저장
        """
        documents = []
        for filename in os.listdir(self.input_folder):
            print(f"Processing: {filename}")
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.input_folder, filename)
                print(f"Processing: {file_path}")

                # 1️⃣ PDF에서 텍스트 추출
                raw_text = self.extract_text_from_pdf(file_path)

                # 2️⃣ 텍스트 정제 및 문장 리스트 변환
                cleaned_text_list = self.clean_text(raw_text)

                # 3️⃣ 리스트 형태로 저장
                documents.extend(cleaned_text_list)

        # 4️⃣ JSON으로 저장
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=4)

        print(f"✅ Processed data saved to {self.output_file}")

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdfs()
