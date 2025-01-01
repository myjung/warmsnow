import anthropic
import json
from typing import Dict, List, Tuple
import time
import logging
import toml
from pathlib import Path
from myutils import read_csv
import os


class GameTranslator:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", save_dir: str = "translation_data"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # 로깅 설정
        self.setup_logging()

        # 저장된 데이터 복구
        self.glossary = self.load_json("glossary.json", {})
        self.translated = self.load_json("translations.json", {})
        self.target_length = 1000  # 초기 목표 텍스트 길이

    def setup_logging(self):
        log_file = self.save_dir / "translation.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_json(self, filename: str, default_value: dict) -> dict:
        file_path = self.save_dir / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error loading {filename}: {e}")
                return default_value
        return default_value

    def save_json(self, data: dict, filename: str):
        file_path = self.save_dir / filename
        # 임시 파일에 먼저 저장
        temp_path = file_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # 성공적으로 저장되면 원본 파일 교체
            temp_path.replace(file_path)
        except Exception as e:
            self.logger.error(f"Error saving {filename}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def create_chunk(self, texts: Dict[str, str]) -> List[Tuple[str, Dict[str, str]]]:
        chunks = []
        current_chunk = {}
        current_length = 0

        for id_, text in texts.items():
            if id_ in self.translated:
                continue

            text_length = len(text)

            # 현재 청크가 목표 길이를 초과하면 새로운 청크 시작
            if current_length + text_length > self.target_length and current_chunk:
                chunks.append(self.prepare_request(current_chunk))
                current_chunk = {}
                current_length = 0

            current_chunk[id_] = text
            current_length += text_length

        if current_chunk:
            chunks.append(self.prepare_request(current_chunk))

        return chunks

    def prepare_request(self, texts: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        # 용어를 길이순으로 정렬하여 중복 체크
        def is_subprocess(short_term: str, term_list: list) -> bool:
            """주어진 용어가 다른 긴 용어의 부분인지 확인"""
            for long_term in term_list:
                if short_term != long_term and short_term in long_term:
                    return True
            return False

        # 현재 glossary의 용어들을 길이 순으로 정렬
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)

        # 더 긴 용어의 부분이 아닌 용어만 선택
        relevant_terms = {}
        for term in sorted_terms:
            # 현재 텍스트들 중 하나라도 이 용어를 포함하고 있는지 확인
            if any(term in text for text in texts.values()):
                # 이미 선택된 더 긴 용어의 부분이 아닌 경우에만 추가
                if not is_subprocess(term, [t for t in relevant_terms.keys()]):
                    relevant_terms[term] = self.glossary[term]

        request_data = {"terms_dictionary": relevant_terms, "texts": texts}

        # 요청 데이터 로깅
        self.logger.info(f"Preparing request with terms dictionary size: {len(relevant_terms)}")
        self.logger.info(f"Full request data: {json.dumps(request_data, ensure_ascii=False)}")

        return json.dumps(request_data, ensure_ascii=False), texts

    def process_response(self, response: list | str, original_texts: Dict[str, str]):
        try:
            if isinstance(response, list):
                response_text = response[0].text if response else ""
            else:
                response_text = response

            self.logger.info(f"Full response received: {response_text}")

            if not response_text:
                self.logger.error("Empty response received")
                return

            result = json.loads(response_text)

            if "result" in result:
                self.translated.update(result["result"])
                self.save_json(self.translated, "translations.json")
                self.logger.info(f"Translated {len(result['result'])} texts")

            if "new_terms" in result:
                new_terms = len(result["new_terms"])
                if new_terms > 0:
                    self.glossary.update(result["new_terms"])
                    self.save_json(self.glossary, "glossary.json")
                    self.logger.info(f"Added new terms to glossary: {result['new_terms']}")

            if "comment" in result:
                self.logger.info(f"Translation comment: {result['comment']}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Error processing response: {e}")
            self.logger.error(f"Raw response: {response_text}")
        except Exception as e:
            self.logger.error(f"Unexpected error processing response: {e}")
            self.logger.error(f"Raw response: {response}")

    def adjust_chunk_size(self, success: bool):
        """응답 성공/실패에 따라 청크 크기 조정"""
        if success:
            self.target_length = min(self.target_length * 1.2, 2400)
        else:
            self.target_length = max(self.target_length * 0.8, 1200)
        self.logger.info(f"Adjusted target length to: {self.target_length}")

    def create_next_chunk(self, texts: Dict[str, str], current_length: int = 0) -> Tuple[str, Dict[str, str]]:
        """다음 청크 생성"""
        current_chunk = {}
        total_length = 0

        for id_, text in texts.items():
            if id_ in self.translated:
                continue

            text_length = len(text)
            if total_length + text_length > self.target_length:
                break

            current_chunk[id_] = text
            total_length += text_length

        if current_chunk:
            return self.prepare_request(current_chunk)
        return None, None

    def translate_all(self, texts: Dict[str, str]):
        total_texts = len(texts)
        processed_count = len(self.translated)
        remaining_texts = {k: v for k, v in texts.items() if k not in self.translated}

        self.logger.info(f"Starting translation: {total_texts} total texts, " f"{processed_count} already translated")

        while remaining_texts:
            request_data, original_chunk = self.create_next_chunk(remaining_texts)
            if not request_data:
                break

            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    self.logger.info(f"Processing chunk size: {len(original_chunk)}")

                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=8192,
                        temperature=0,
                        system='You are a Chinese to Korean translator specializing in wuxia game localization. Follow these guidelines for translation:\n\n1. Markup and placeholder handling:\n- Keep {placeholder} unchanged and add appropriate Korean particles\n  e.g.) "{player_name}获得了{item_count}个{item_name}" \n        → "{player_name}이(가) {item_name}을(를) {item_count}개 획득했습니다"\n- Preserve <tag></tag> structure and only translate the text inside\n  e.g.) "<color>获得{item_count}个{item_name}</color>" \n        → "<color>{item_name}을(를) {item_count}개 획득했습니다</color>"\n\n2. Translation principles:\n- Consider meaning and rhythm when translating faction names, character names, and martial arts skills\n- Translate UI concisely and story text with appropriate style\n- Follow Korean wuxia conventions for genre terms (기공, 내공, etc.)\n- Keep commonly used Chinese idioms in Korean hanja form, localize unfamiliar ones\n- Choose contextually appropriate translations for single-character words\n  e.g.) "魂" can be "혼" or "넋" depending on context\n  e.g.) "气" can be "기" or "공기" depending on context\n\n3. Terms extraction:\n- Add source text and translated terms to new_terms:\n  * Complete item and skill names\n  * Full character/faction names \n  * System features and mechanics\n  * Recurring multi-character game terms\n  * Special effects and states\n- Do NOT add:\n  * Single characters that form parts of longer terms\n  * Generic single-character words with multiple contextual meanings\n\n4. Input/Output format:\nInput: {\n    "terms_dictionary": {"chinese": "korean"},     // optional, existing translations\n    "texts": {"id": "source text"}                // required\n}\n\nOutput: {\n    "result": {"id": "translated text with escaped quotes"},  \n    "comment": "review notes",                    // optional, only include if review or clarification needed\n    "new_terms": {"chinese": "korean"}            // only add complete terms, not components\n}\n\nNote: Always escape quotes in text content with backslash: \\"example\\"',
                        messages=[{"role": "user", "content": [{"type": "text", "text": request_data}]}],
                    )

                    # 응답 처리 전 전체 응답 로깅
                    if isinstance(message.content, list):
                        response_text = message.content[0].text if message.content else ""
                    else:
                        response_text = message.content
                    self.logger.info(f"Raw API response: {response_text}")

                    self.process_response(message.content, original_chunk)
                    self.adjust_chunk_size(True)

                    # 번역된 텍스트 제거
                    remaining_texts = {k: v for k, v in texts.items() if k not in self.translated}

                    processed_count = len(self.translated)
                    progress = (processed_count / total_texts) * 100
                    self.logger.info(f"Progress: {progress:.2f}% ({processed_count}/{total_texts})")
                    self.logger.info(f"Current glossary size: {len(self.glossary)}")

                    time.sleep(1)
                    break

                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"Error processing chunk (attempt {retry_count}/{max_retries}): {e}")
                    self.adjust_chunk_size(False)

                    if retry_count == max_retries:
                        self.logger.error(f"Failed to process chunk after {max_retries} attempts")
                        # 실패한 청크의 텍스트들을 기록
                        self.save_json(original_chunk, "failed_translations.json")
                    else:
                        time.sleep(retry_count * 2)

        self.logger.info("Translation completed")
        return self.translated


# 사용 예시
if __name__ == "__main__":
    from pprint import pprint

    configfile = toml.load("localconfig.toml")
    api_key = configfile["local"]["CLAUDE_API_KEY"]
    data = read_csv("1230.csv")
    translator = GameTranslator(api_key)

    need_trans_name_start = (
        "PN",
        "Skill",
        "TimeEcho",
        "TimeTalent",
    )
    texts_to_translate = {row["Name"]: row["CHT"] for row in data}
    # texts_to_translate = {
    #     k: v for k, v in texts_to_translate.items() if any(k.startswith(y) for y in need_trans_name_start)
    # }
    # pprint(texts_to_translate)
    translated_texts = translator.translate_all(texts_to_translate)
