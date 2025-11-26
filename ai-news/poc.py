import os
import json
import feedparser
import requests

from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# .env 파일 로드
load_dotenv()

KOREAN_NEWS_DOMAINS = [
    "yna.co.kr",        # 연합뉴스
    "hani.co.kr",       # 한겨레
    "chosun.com",       # 조선일보
    "joongang.co.kr",   # 중앙일보
    "donga.com",        # 동아일보
    "khan.co.kr",       # 경향신문
    "kmib.co.kr",       # 국민일보
    "news.kbs.co.kr",   # KBS 뉴스
    "imnews.imbc.com",  # MBC 뉴스
    "news.sbs.co.kr",   # SBS 뉴스
    "ytn.co.kr",        # YTN
    "newsis.com",       # 뉴시스
    "news1.kr",         # 뉴스1
]

def get_today_trend_topics():
    print("\n📡 [Google News] 대한민국 실시간 주요 뉴스를 스캔하고 있습니다...")
    
    # 구글 뉴스 RSS URL (대한민국/한국어 설정)
    # ceid=KR:ko -> 한국 지역, 한국어
    rss_url = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
    
    try:
        feed = feedparser.parse(rss_url)
        entries = feed.entries[:20] # 상위 20개만 가져옴
    except Exception as e:
        print(f"RSS 파싱 실패: {e}")
        return []

    if not entries:
        print("뉴스를 찾을 수 없습니다.")
        return []
    
    # 제목 리스트 추출
    headlines = [f"- {entry.title}" for entry in entries]
    headlines_text = "\n".join(headlines)
    
    print(f"✅ {len(headlines)}개의 헤드라인 확보 완료!")

    print("🧠 AI 편집장이 카드뉴스 주제를 선별하고 있습니다...")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template("""
    아래는 방금 수집된 '구글 뉴스(대한민국)'의 헤드라인 리스트야.
    이 중에서 카드뉴스로 제작하기 가장 좋은 **'정치/사회적 논쟁 이슈'** 10가지를 뽑아줘.

    [헤드라인 리스트]
    {headlines}

    [요구사항]
    1. '단순 사건사고(화재, 날씨)'나 '스포츠/연예'는 제외할 것.
    2. '정책 찬반', '국회 갈등', '사회적 법안' 등 논쟁거리가 있는 주제 우선.
    3. 구체적인 키워드 문장으로 출력.
    
    [출력 포맷 - JSON 리스트]
    ["의대 정원 증원 갈등", "국민연금 개혁안 논란", "전세사기 특별법 개정"]
    """)

    chain = prompt | llm | parser
    
    try:
        topics = chain.invoke({"headlines": headlines_text})
        return topics
    except Exception as e:
        print(f"토픽 추출 실패: {e}")
        return []

def run_news_processor(topic):
    # ---------------------------------------------------------
    # 1. 뉴스 검색 (Data Collection)
    # ---------------------------------------------------------
    
    today_str = datetime.now().strftime("%Y년 %m월 %d일")
    print(f"🔍 {today_str} '{topic}' 관련 최신 기사를 검색 중입니다 (DuckDuckGo)...")
    
    article_url = None
    article_title = topic
    
    try:
        # region='kr-kr' : 한국 리전 강제 설정
        ddgs = DDGS()
        results = ddgs.text(f"{topic}", region='kr-kr', max_results=3)
        
        if results:
            # 첫 번째 검색 결과의 링크를 가져옵니다.
            best_match = results[0]
            article_url = best_match['href']
            article_title = best_match['title']
            print(f"✅ 기사 발견: {article_title}")
            print(f"🔗 링크: {article_url}")
        else:
            return {"error": "관련된 한국 기사를 찾을 수 없습니다."}
            
    except Exception as e:
        return {"error": f"검색 중 오류 발생: {e}"}

    # 2. Jina Reader로 본문 스크랩 (가장 깔끔한 방법)
    # URL 앞에 'https://r.jina.ai/'만 붙이면 AI용 텍스트로 변환해줍니다.
    print("📖 기사 본문을 읽어오는 중입니다 (Jina Reader)...")
    try:
        jina_url = f"https://r.jina.ai/{article_url}"
        response = requests.get(jina_url)
        full_content = response.text
        
        # 내용이 너무 길면 자름 (토큰 절약)
        if len(full_content) > 5000:
            full_content = full_content[:5000] + "...(생략)"
            
    except Exception as e:
        return {"error": f"본문 스크랩 실패: {e}"}

    # 3. AI 분석 및 데이터 구조화
    print("🤖 AI가 카드뉴스 데이터를 생성하고 있습니다...")

    # temperature=0 : 창의성보다는 정확한 포맷 준수를 위해 0으로 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template("""
    너는 정치/사회 이슈를 일반 시민들이 이해하기 쉽게 설명해주는 '친절한 에디터'야.
    아래 뉴스 기사를 읽고, 모바일 카드뉴스 앱에서 사용할 수 있는 데이터를 **반드시 JSON 형식**으로 추출해줘.
    
    [기사 내용]
    {content}
    
    [오늘 날짜]
    {today}

    [요구사항]
    1. 대상 독자: 정치/사회 이슈를 잘 모르는 초심자
    2. 어조: 객관적이지만 친절하고 쉬운 해요체
    3. **출력 포맷(JSON)을 엄격히 지킬 것.**

    [JSON 출력 스키마]
    {{
        "card_title": "카드뉴스 표지 제목 (30자 이내, 클릭하고 싶게)",
        "summary_cards": [
            "배경: 사건의 발단이나 배경 (핵심 요약)",
            "쟁점: 주요 쟁점이나 현재 상황",
            "전망: 앞으로의 전망이나 영향"
        ],
        "vote_guide": {{
            "question": "찬반 투표 질문 (예: '의대 증원에 찬성하나요?')",
            "pros": "찬성 측의 핵심 논리 (단답형). 예: 찬성, 의료 인력 충원",
            "cons": "반대 측의 핵심 논리 (단답형 ). 예: 반대, 의료 질 저하"
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
            "content": full_content,
            "today": today_str
        })
        
        # 원본 링크 정보 추가 (나중에 앱에서 '원문 보기' 버튼에 씀)
        result_json['original_url'] = article_url
        
        return result_json
        
    except Exception as e:
        return {"error": f"AI 처리 중 오류 발생: {e}"}

# ---------------------------------------------------------
# 실행부 (터미널에서 python main.py로 실행 시 작동)
# ---------------------------------------------------------
if __name__ == "__main__":
    
    suggested_topics = get_today_trend_topics()
    
    if suggested_topics:
        print("\n" + "="*30)
        print("📢 AI 편집장이 추천하는 오늘의 이슈")
        print("="*30)
        for idx, topic in enumerate(suggested_topics):
            print(f"{idx + 1}. {topic}")
        print("0. 직접 입력하기")
        print("="*30)

        choice = input("\n번호를 선택하세요: ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(suggested_topics):
            selected_topic = suggested_topics[int(choice) - 1]
        else:
            selected_topic = input("주제를 직접 입력하세요: ")
    else:
        # 추천 실패 시 바로 입력 모드
        print("추천된 주제가 없습니다.")
        selected_topic = input("주제를 직접 입력하세요: ")

    # 2단계: 선택된 주제로 심층 분석 실행
    final_data = run_news_processor(selected_topic)

    print("\n" + "="*20 + " [최종 결과] " + "="*20)
    print(json.dumps(final_data, indent=2, ensure_ascii=False))