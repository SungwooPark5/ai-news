import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# .env 파일에서 환경 변수 로드
load_dotenv()

# 1. 설정 (API 키 필요)
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def summarize_news(topic):
    # 2. 검색 도구 초기화 (최신 기사 3개 검색)
    search = TavilySearch(k=3)

    print(f"🔍 '{topic}' 관련 기사를 검색 중입니다...")
    try:
        search_results = search.invoke(topic)
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {e}"
    
    # 3. 검색된 내용 합치기 (Tavily는 URL과 함께 요약된 내용도 일부 반환함)
    context_text = ""
    for item in search_results['results']:
        context_text += f"출처: {item['url']}\n내용: {item['content']}\n\n"

    # 4. LLM 프롬프트 구성
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_template("""
    아래 제공된 뉴스 기사들을 바탕으로 '{topic}'에 대한 최신 동향을 요약해줘.
    
    [요구사항]
    1. 각 기사의 핵심 내용을 통합해서 3문항으로 나열하여 서술할 것.
    2. 전문적인 톤앤매너를 유지할 것.
    3. 마지막에 원본 출처 링크를 리스트로 제공할 것.
    4. 불필요한 수식어나 장황한 설명은 피할 것.
    5. 최신 정보에 집중할 것.
    
    [기사 데이터]
    {context}
    """)
    
    # 5. 실행 및 결과 반환
    print("🤖 AI가 요약 보고서를 작성 중입니다...")
    chain = prompt | llm
    summary = chain.invoke({"topic": topic, "context": context_text})
    
    return summary.content

# 실행
if __name__ == "__main__":
    user_input = input("요약할 뉴스 주제를 입력하세요: ")
    result = summarize_news(user_input)
    print("\n" + "="*50)
    print(result)
    print("="*50)