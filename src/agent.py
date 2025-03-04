import os
import json 
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class RAGEvaluator:
    def __init__(self, db_path="data/vector_db", model="gpt-4o-mini", use_optimization=False):
        """
        RAG 기반 QA + KMMLU 평가 통합 클래스

        Args:
        - db_path (str): FAISS 벡터 DB 경로
        - model (str): OpenAI LLM 모델 (기본값: "gpt-4o-mini")
        - use_optimization (bool): Self-Consistency 적용 여부 (기본값: False)
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_optimization = use_optimization

        # 🔹 OpenAI LLM 설정
        self.llm = ChatOpenAI(model=model, openai_api_key=self.api_key)

        # 🔹 FAISS 벡터 DB 로드
        self.vectorstore = FAISS.load_local(
            db_path,
            OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self.api_key),
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()

        # 🔹 KMMLU 데이터 로드
        self.dataset = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law")["test"]

    def retrieve_documents(self, query, top_k=3):
        """
        벡터 검색을 수행하여 관련 문서 반환

        Args:
        - query (str): 사용자 질문
        - top_k (int): 검색할 문서 개수 (기본값: 3)

        Returns:
        - str: 검색된 문서 텍스트 (결합된 형태)
        """
        related_docs = self.retriever.invoke(query)
        retrieved_texts = "\n".join([doc.page_content for doc in related_docs[:top_k]])
        return retrieved_texts

    def format_kmmlu_question(self, example):
        """
        KMMLU 데이터셋을 GPT가 처리할 수 있는 형식으로 변환

        Args:
        - example (dict): KMMLU 문제 데이터

        Returns:
        - str: GPT에게 전달할 프롬프트 형식의 질문
        """
        question = example['question']
        options = [f"({i+1}) {example[opt]}" for i, opt in enumerate(["A", "B", "C", "D"])]
        formatted_text = f"{question}\n" + "\n".join(options)
        return formatted_text

    def generate_answer(self, example):
        """
        벡터 검색 후 GPT-4o-mini를 사용하여 4지 선다형 답변 생성

        Args:
        - example (dict): KMMLU 문제 데이터

        Returns:
        - str: GPT가 선택한 정답 (1, 2, 3, 4 중 하나)
        """
        formatted_question = self.format_kmmlu_question(example)
        retrieved_texts = self.retrieve_documents(example["question"])

        prompt = f"""
        🔍 참고 문서:
        {retrieved_texts}

        ❓ 질문:
        {formatted_question}

        🎯 위 질문에 대해 주어진 선택지 중에서 가장 적절한 답변을 **단 하나의 숫자 (1, 2, 3, 4)로만** 출력하세요.
        """
        prompt = prompt.strip()
        response = self.llm.invoke(prompt).content.strip()

        # ✅ GPT가 응답한 값이 1, 2, 3, 4 중 하나인지 확인 후 반환
        valid_choices = {"1", "2", "3", "4"}
        return response if response in valid_choices else "N/A"

    def evaluate_sample(self, index=0):
        """
        특정 KMMLU 문제 하나를 평가하고 검색 결과 및 응답을 출력

        Args:
        - index (int): 평가할 문제의 인덱스 (기본값: 0)
        """
        example = self.dataset[index]
        retrieved_texts = self.retrieve_documents(example["question"])
        formatted_question = self.format_kmmlu_question(example)

        print("\n🔍 **검색된 문서** 🔍")
        print(retrieved_texts)

        print("\n❓ **KMMLU 문제 (포맷된 질문)** ❓")
        print(formatted_question)

        # GPT 응답 얻기
        answer = self.generate_answer(example)
        correct_answer = str(example["answer"])

        print("\n🤖 **GPT의 선택**:", answer)
        print("✅ **정답 (KMMLU 제공)**:", correct_answer)

        # 정답 여부 확인
        if answer == correct_answer:
            print("\n🎯 **정답!** ✅")
        else:
            print("\n❌ **오답!** ❌")

    def evaluate_kmmlu(self):
        """
        KMMLU Criminal-Law 데이터셋을 활용하여 평가 수행

        Returns:
        - float: 최종 Accuracy
        """
        responses_per_question = []
        num_iterations = 5 if self.use_optimization else 1  # Self-Consistency 적용 여부

        # for example in tqdm(self.dataset, desc="Processing KMMLU Evaluation"):
        for idx, example in enumerate(tqdm(self.dataset, desc="Processing KMMLU Evaluation", disable=False)): 
            print(f"📝 [{idx+1}/{len(self.dataset)}] 질문 처리 중: {example['question']}")  
            question_responses = []

            for _ in range(num_iterations):
                answer = self.generate_answer(example)
                question_responses.append(answer.strip())

            # Self-Consistency 적용: 최빈값 선택
            final_answer = Counter(question_responses).most_common(1)[0][0] if self.use_optimization else question_responses[0]
            responses_per_question.append(final_answer)

        # 정답 비교 및 Accuracy 계산
        correct_answers = [str(example["answer"]) for example in self.dataset]  # KMMLU 정답은 숫자 문자열
        accuracy = sum(1 for gt, pred in zip(correct_answers, responses_per_question) if gt == pred) / len(correct_answers) * 100

        # 평가 결과 저장
        score_path = f"output/kmmlu_score{'_optimized' if self.use_optimization else ''}.txt"
        # ✅ 출력 폴더가 없으면 생성
        os.makedirs(os.path.dirname(score_path), exist_ok=True)
        with open(score_path, "w") as f:
            f.write(f"KMMLU Criminal-Law Score: {accuracy:.2f}%\n")

        print(f"🎯 KMMLU Criminal-Law Score: {accuracy:.2f}% (Optimized: {self.use_optimization})")
        return accuracy

    def save_batch_io(self, batch_input_path="batch_input.jsonl", batch_output_path="batch_output.jsonl"):
        """
        KMMLU 데이터셋 전체에 대해 batch API에 사용된 input (프롬프트 등)과 
        output (모델의 응답)을 JSONL 파일로 저장합니다.

        Args:
        - batch_input_path (str): 저장할 batch input JSONL 파일 경로
        - batch_output_path (str): 저장할 batch output JSONL 파일 경로
        """
        # 출력 디렉토리가 없다면 생성
        os.makedirs(os.path.dirname(batch_input_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(batch_output_path) or ".", exist_ok=True)

        with open(batch_input_path, "w", encoding="utf-8") as fin, open(batch_output_path, "w", encoding="utf-8") as fout:
            for example in tqdm(self.dataset, desc="Saving batch IO"):
                formatted_question = self.format_kmmlu_question(example)
                retrieved_texts = self.retrieve_documents(example["question"])
                prompt = f"""
                🔍 참고 문서:
                {retrieved_texts}

                ❓ 질문:
                {formatted_question}

                🎯 위 질문에 대해 주어진 선택지 중에서 가장 적절한 답변을 **단 하나의 숫자 (1, 2, 3, 4)로만** 출력하세요.
                """
                prompt = prompt.strip()

                # 모델 호출 및 응답
                response = self.llm.invoke(prompt).content.strip()
                valid_choices = {"1", "2", "3", "4"}
                answer = response if response in valid_choices else "N/A"

                # batch input 정보 구성
                input_data = {
                    "question": example["question"],
                    "formatted_question": formatted_question,
                    "retrieved_documents": retrieved_texts,
                    "prompt": prompt
                }
                # batch output 정보 구성
                output_data = {
                    "response": answer
                }
                # JSONL 파일에 한 줄씩 기록
                fin.write(json.dumps(input_data, ensure_ascii=False) + "\n")
                fout.write(json.dumps(output_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    evaluator = RAGEvaluator(use_optimization=True)  # Self-Consistency 사용

    # ✅ 특정 데이터 샘플 평가
    evaluator.evaluate_sample(index=0)

    # ✅ 전체 KMMLU 데이터셋 평가
    evaluator.evaluate_kmmlu()

    # ✅ batch input, output 정보를 JSONL 파일로 저장
    evaluator.save_batch_io(batch_input_path="output/batch_input.jsonl", batch_output_path="output/batch_output.jsonl")
