import os
import json
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# .env 파일 로드
load_dotenv()

def run_news_processor(topic):
    # ---------------------------------------------------------
    # 1. 뉴스 검색 (Data Collection)
    # ---------------------------------------------------------
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"🔍 {today_str} '{topic}' 관련 최신 기사를 검색 중입니다...")
    
    # LangChain Tool 대신 TavilyClient를 직접 사용 (데이터 구조 안정성 위함)
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    try:
        response = tavily.search(
            query=topic,
            topic="news",
            days=1,       # 최근 1일 이내 기사만
            max_results=5 # 상위 5개만 조회
        )
        results = response.get('results', [])
    except Exception as e:
        return {"error": f"검색 중 오류 발생: {e}"}

    if not results:
        print("⚠️ 24시간 이내에 관련 기사가 없어 3일치로 범위를 넗힙니다...")
        try:
            response = tavily.search(
                query=topic,
                topic="news",
                days=3,       # 최근 3일 이내 기사로 확장
                max_results=5
            )
            results = response.get('results', [])
        except Exception as e:
            return {"error": f"기사를 찾을 수 없습니다."}
        
    if not results:
        return {"error": "관련 기사를 찾을 수 없습니다."}

    # 프로토타입용: 가장 관련성 높은 첫 번째 기사 선택
    target_news = results[0]
    print(f"✅ 선택된 기사: {target_news['title']}")
    print(f"🔗 링크: {target_news['url']}\n")

    # ---------------------------------------------------------
    # 2. AI 분석 및 데이터 구조화 (AI Processing)
    # ---------------------------------------------------------
    print("🤖 AI가 카드뉴스 데이터와 퀴즈를 생성하고 있습니다...")

    # temperature=0 : 창의성보다는 정확한 포맷 준수를 위해 0으로 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template("""
    너는 정치/사회 이슈를 일반 시민들이 이해하기 쉽게 설명해주는 '친절한 에디터'야.
    아래 뉴스 기사를 읽고, 모바일 카드뉴스 앱에서 사용할 수 있는 데이터를 **반드시 JSON 형식**으로 추출해줘.

    [기사 정보]
    제목: {title}
    내용: {content}
    작성일: {today}

    [요구사항]
    1. 대상 독자: 정치/사회 이슈를 잘 모르는 초심자
    2. 어조: 객관적이지만 친절하고 쉬운 해요체
    3. **출력 포맷(JSON)을 엄격히 지킬 것.**

    [JSON 출력 스키마]
    {{
        "card_title": "카드뉴스 표지 제목 (30자 이내, 클릭하고 싶게)",
        "summary_cards": [
            "카드1: 사건의 발단이나 배경 (핵심 요약)",
            "카드2: 주요 쟁점이나 현재 상황",
            "카드3: 앞으로의 전망이나 영향"
        ],
        "vote_guide": {{
            "question": "찬반 투표 질문 (예: '의대 증원에 찬성하십니까?')",
            "pros": "찬성 측의 핵심 논리 (한두 단어). 예: 찬성, 의료 인력 충원",
            "cons": "반대 측의 핵심 논리 (한두 단어 ). 예: 반대, 의료 질 저하"
        }},
        "quiz": {{
            "question": "기사 내용을 확인하는 객관식 퀴즈 (난이도 하)",
            "options": ["보기1", "보기2", "보기3", "보기4"],
            "answer_index": 0 (정답 보기의 인덱스 0~3, 숫자만 반환)
        }}
    }}
    """)

    # 체인 연결
    chain = prompt | llm | parser

    try:
        result_json = chain.invoke({
            "title": target_news['title'],
            "content": target_news['content'],
            "today": today_str
        })
        
        # 원본 링크 정보 추가 (나중에 앱에서 '원문 보기' 버튼에 씀)
        result_json['original_url'] = target_news['url']
        
        return result_json
        
    except Exception as e:
        return {"error": f"AI 처리 중 오류 발생: {e}"}

# ---------------------------------------------------------
# 실행부 (터미널에서 python main.py로 실행 시 작동)
# ---------------------------------------------------------
if __name__ == "__main__":
    user_input = input("주제를 입력하세요 (예: 의대 증원): ")
    final_data = run_news_processor(user_input)

    print("\n" + "="*20 + " [최종 JSON 데이터] " + "="*20)
    # 한글 깨짐 방지를 위해 ensure_ascii=False 사용
    print(json.dumps(final_data, indent=2, ensure_ascii=False))