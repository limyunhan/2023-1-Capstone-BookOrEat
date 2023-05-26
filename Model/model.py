import pandas as pd
import json
import re
from transformers import BertTokenizer

# 데이터 파일 경로
review_file = "review.csv"
user_review_file = "user_review_id.json"
review_content_file = "review.json"
menu_file = "menu_json.json"

# 데이터 로드
review_data = pd.read_csv(review_file)
user_review_data = json.load(open(user_review_file))
review_content_data = json.load(open(review_content_file))
menu_data = json.load(open(menu_file))

# 데이터 전처리
review_data = review_data.dropna(subset=["Timestamp"])  # 리뷰 없는 데이터 제거

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 불용어 처리 함수
def remove_stopwords(text):
    stopwords = ["a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for"]
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)

# 텍스트 정제 및 정규화 함수
def clean_text(text):
    # 소문자로 변환
    text = text.lower()
    # 특수 문자 제거
    text = re.sub(r"[^a-zA-Z0-9가-힣\s]", "", text)
    # 중복 공백 제거
    text = re.sub(r"\s+", " ", text)
    # 불용어 처리
    text = remove_stopwords(text)
    return text

# 각 단계별 예시 코드
# 라벨링
review_data["Label"] = review_data["Sentiment"].apply(lambda x: 1 if x == "positive" else 0)

# 데이터셋 분할
train_ratio = 0.8
train_size = int(len(review_data) * train_ratio)
train_data = review_data[:train_size]
valid_data = review_data[train_size:]

# BERT 입력 형식 변환
def convert_text_to_bert_inputs(text):
    # 텍스트 정제 및 정규화
    text = clean_text(text)
    encoded_inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,  # 적절한 최대 길이로 설정
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoded_inputs

train_data["BERTInputs"] = train_data["Review"].apply(convert_text_to_bert_inputs)
valid_data["BERTInputs"] = valid_data["Review"].apply(convert_text_to_bert_inputs)

# 메뉴 데이터 전처리
for restaurant_name, menu in menu_data.items():
    # 메뉴 이름과 가격 추출
    menu_list = []
    for item in menu.split("\n"):
        item = item.strip()
        if item != "":
            menu_name, price = item.split()
            menu_list.append({"menu_name": menu_name, "price": price})
    menu_data[restaurant_name] = menu_list
