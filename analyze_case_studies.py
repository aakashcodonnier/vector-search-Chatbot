#!/usr/bin/env python3
"""
Analyze case study page HTML structure
"""

from bs4 import BeautifulSoup

def analyze_html():
    with open('case_study_page.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    articles = soup.find_all('article')
    
    print(f"Total articles found: {len(articles)}")
    print("\nArticle details:")
    
    for i, article in enumerate(articles[:5], 1):  # Show first 5
        print(f"\n{i}. ARTICLE ANALYSIS:")
        
        # Find title
        title_h1 = article.find('h1', class_='entry-title')
        title_h2 = article.find('h2', class_='entry-title')
        title_any_h1 = article.find('h1')
        title_any_h2 = article.find('h2')
        
        title = title_h1 or title_h2 or title_any_h1 or title_any_h2
        if title:
            print(f"   Title: {title.get_text(strip=True)[:100]}...")
        
        # Find link
        link = title.find('a') if title else article.find('a')
        if link:
            print(f"   Link: {link.get('href', 'N/A')}")
        
        # Check for content div
        content_div = (article.find('div', class_='entry-content') or 
                      article.find('div', class_='post-content'))
        if content_div:
            print(f"   Has content div: YES")
        
        # Check categories
        cats = (article.select('.cat-links a') or 
               article.select('.categories a') or 
               article.select('.tags a'))
        if cats:
            cat_list = [c.get_text(strip=True) for c in cats]
            print(f"   Categories: {cat_list}")

if __name__ == "__main__":
    analyze_html()