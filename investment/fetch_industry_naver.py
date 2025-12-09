"""Fetch industry classification from Naver Finance for KOSPI stocks.

Usage:
    python fetch_industry_naver.py --data-dir ./kospi --output ./kospi_industry.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


def get_industry_mapping() -> Dict[int, str]:
    """Fetch industry code to name mapping from Naver PC site."""
    url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        links = soup.select('a[href*="sise_group_detail"]')
        industry_map = {}
        for link in links:
            href = link.get("href", "")
            if "no=" in href:
                code = href.split("no=")[1].split("&")[0]
                name = link.get_text().strip()
                if code.isdigit() and name:
                    industry_map[int(code)] = name
        
        if industry_map:
            return industry_map
    except Exception as e:
        print(f"Failed to get industry mapping from PC site: {e}")
    
    # Fallback: hardcoded complete mapping
    return {
        273: "자동차", 279: "건설", 306: "전기장비", 307: "전자제품",
        295: "에너지장비및서비스", 322: "비철금속", 270: "자동차부품",
        284: "우주항공과국방", 325: "전기유틸리티", 327: "디스플레이패널",
        329: "도로와철도운송", 283: "전기제품", 315: "손해보험", 291: "조선",
        278: "반도체와반도체장비", 323: "해운사", 304: "철강", 277: "창업투자",
        300: "양방향미디어와서비스", 320: "건축제품", 272: "화학", 301: "은행",
        303: "가구", 289: "건축자재", 326: "항공화물운송과물류",
        302: "식품과기본식료품소매", 321: "증권", 298: "가정용기기와용품",
        266: "화장품", 305: "항공사", 330: "생명보험", 332: "문구류",
        334: "무역회사와판매업체", 267: "IT서비스", 339: "다각화된소비자서비스",
        312: "가스유틸리티", 276: "복합기업", 328: "전문소매",
        324: "상업서비스와공급품", 299: "기계", 331: "복합유틸리티",
        333: "무선통신서비스", 280: "부동산", 263: "게임엔터테인먼트",
        265: "판매업체", 313: "석유와가스", 337: "카드", 282: "전자장비와기기",
        294: "통신장비", 296: "운송인프라", 336: "다각화된통신서비스",
        317: "호텔,레스토랑,레저", 269: "디스플레이장비및부품",
        293: "컴퓨터와주변기기", 310: "광고", 264: "백화점과일반상점",
        275: "담배", 274: "섬유,의류,신발,호화품", 297: "가정용품",
        318: "종이와목재", 292: "핸드셋", 268: "식품", 314: "출판",
        311: "포장재", 290: "교육서비스", 287: "소프트웨어",
        281: "건강관리장비와용품", 271: "레저용장비와제품", 319: "기타금융",
        262: "생명과학도구및서비스", 309: "음료", 285: "방송과엔터테인먼트",
        316: "건강관리업체및서비스", 308: "인터넷과카탈로그소매",
        288: "건강관리기술", 338: "사무용전자제품", 261: "제약", 286: "생물공학"
    }


def get_stock_industry(symbol: str, industry_map: Dict[int, str]) -> tuple[Optional[str], Optional[int]]:
    """Fetch industry info for a single stock.
    
    Returns:
        (industry_name, industry_code) or (None, None) if not found
    """
    url = f"https://m.stock.naver.com/api/stock/{symbol}/integration"
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 10)"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            industry_code = data.get("industryCode")
            if industry_code:
                code = int(industry_code)
                name = industry_map.get(code, f"Unknown_{code}")
                return name, code
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
    
    return None, None


def fetch_all_industries(data_dir: Path, delay: float = 0.2) -> pd.DataFrame:
    """Fetch industry info for all symbols in data_dir.
    
    Args:
        data_dir: Directory containing <symbol>.csv files
        delay: Delay between requests in seconds
        
    Returns:
        DataFrame with columns: symbol, industry, industry_code
    """
    # First, get the industry code -> name mapping
    print("Fetching industry mapping from Naver...")
    industry_map = get_industry_mapping()
    print(f"Found {len(industry_map)} industries")
    
    csv_files = sorted(data_dir.glob("*.csv"))
    symbols = [f.stem for f in csv_files]
    
    print(f"Found {len(symbols)} symbols in {data_dir}")
    
    results = []
    for i, symbol in enumerate(symbols):
        industry, code = get_stock_industry(symbol, industry_map)
        
        if industry:
            print(f"[{i+1}/{len(symbols)}] {symbol}: {industry} (code={code})")
        else:
            print(f"[{i+1}/{len(symbols)}] {symbol}: NOT FOUND")
            industry = "Unknown"
            code = -1
        
        results.append({
            "symbol": symbol,
            "industry": industry,
            "industry_code": code
        })
        
        # Rate limiting
        if i < len(symbols) - 1:
            time.sleep(delay)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Fetch industry classification from Naver Finance")
    parser.add_argument("--data-dir", type=str, default="./kospi", help="Directory with stock CSVs")
    parser.add_argument("--output", type=str, default="./kospi_industry.csv", help="Output CSV path")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between requests (seconds)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    df = fetch_all_industries(data_dir, delay=args.delay)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    
    print(f"\nSaved to {args.output}")
    print(f"\nIndustry distribution:")
    print(df["industry"].value_counts())


if __name__ == "__main__":
    main()
