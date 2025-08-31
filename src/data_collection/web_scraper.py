"""
Web Scraper for EV Charging Station Data Collection
Handles automated data extraction from various EV charging websites
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from playwright.async_api import async_playwright
import pandas as pd

class WebScraper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('user_agent', 'Mozilla/5.0 (compatible; EV-Research-Bot/1.0)')
        })
        self.scraped_data = []
        self.visited_urls = set()
        self.delay = config.get('delay_between_requests', 2.0)
        self.max_pages = config.get('max_pages', 1000)
        self.allowed_domains = config.get('allowed_domains', [])
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def scrape_ev_websites(self) -> List[Dict[str, Any]]:
        """Main method to scrape EV charging websites"""
        self.logger.info("Starting EV website scraping")
        
        for domain in self.allowed_domains:
            try:
                self.logger.info(f"Scraping domain: {domain}")
                domain_data = await self._scrape_domain(domain)
                self.scraped_data.extend(domain_data)
                
                # Respect rate limiting
                await asyncio.sleep(self.delay)
                
            except Exception as e:
                self.logger.error(f"Error scraping domain {domain}: {e}")
                continue
        
        self.logger.info(f"Scraping completed. Total pages scraped: {len(self.scraped_data)}")
        return self.scraped_data
    
    async def _scrape_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Scrape a specific domain for EV charging information"""
        domain_data = []
        
        # Start with main page
        main_url = f"https://{domain}"
        
        try:
            # Use Playwright for dynamic content
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent
                await page.set_extra_http_headers({
                    'User-Agent': self.config.get('user_agent')
                })
                
                # Navigate to main page
                await page.goto(main_url, wait_until='networkidle')
                
                # Extract main page content
                main_content = await self._extract_page_content(page)
                if main_content:
                    main_content['source_url'] = main_url
                    main_content['domain'] = domain
                    domain_data.append(main_content)
                
                # Find and scrape sub-pages
                sub_pages = await self._find_sub_pages(page, domain)
                
                for sub_page in sub_pages[:self.max_pages // len(self.allowed_domains)]:
                    try:
                        await page.goto(sub_page, wait_until='networkidle')
                        page_content = await self._extract_page_content(page)
                        
                        if page_content:
                            page_content['source_url'] = sub_page
                            page_content['domain'] = domain
                            domain_data.append(page_content)
                        
                        await asyncio.sleep(self.delay)
                        
                    except Exception as e:
                        self.logger.warning(f"Error scraping sub-page {sub_page}: {e}")
                        continue
                
                await browser.close()
                
        except Exception as e:
            self.logger.error(f"Error with Playwright for domain {domain}: {e}")
            # Fallback to requests + BeautifulSoup
            try:
                response = self.session.get(main_url, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content = self._extract_content_bs4(soup, main_url, domain)
                    if content:
                        domain_data.append(content)
            except Exception as fallback_error:
                self.logger.error(f"Fallback scraping also failed for {domain}: {fallback_error}")
        
        return domain_data
    
    async def _extract_page_content(self, page) -> Optional[Dict[str, Any]]:
        """Extract content from a Playwright page"""
        try:
            # Wait for content to load
            await page.wait_for_load_state('networkidle')
            
            # Extract text content
            text_content = await page.evaluate("""
                () => {
                    const body = document.body;
                    if (!body) return null;
                    
                    // Remove script and style elements
                    const scripts = body.querySelectorAll('script, style, nav, footer, header');
                    scripts.forEach(el => el.remove());
                    
                    // Extract main content
                    const main = body.querySelector('main') || body.querySelector('#content') || body.querySelector('.content');
                    const content = main || body;
                    
                    return {
                        title: document.title || '',
                        text: content.innerText || content.textContent || '',
                        html: content.innerHTML || '',
                        url: window.location.href
                    };
                }
            """)
            
            if not text_content:
                return None
            
            # Process and clean content
            processed_content = self._process_content(text_content)
            return processed_content
            
        except Exception as e:
            self.logger.warning(f"Error extracting page content: {e}")
            return None
    
    def _extract_content_bs4(self, soup: BeautifulSoup, url: str, domain: str) -> Optional[Dict[str, Any]]:
        """Extract content using BeautifulSoup (fallback method)"""
        try:
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ''
            
            # Extract main content
            main_content = soup.find('main') or soup.find(id='content') or soup.find(class_='content')
            if main_content:
                main_text = main_content.get_text(separator=' ', strip=True)
            else:
                main_text = text_content
            
            content = {
                'title': title_text,
                'text': main_text,
                'html': str(main_content) if main_content else '',
                'url': url,
                'domain': domain
            }
            
            return self._process_content(content)
            
        except Exception as e:
            self.logger.warning(f"Error extracting content with BeautifulSoup: {e}")
            return None
    
    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean extracted content"""
        # Clean text content
        text = content.get('text', '')
        if text:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove very short content
            if len(text) < 100:
                return None
        
        # Extract EV-related keywords
        ev_keywords = self._extract_ev_keywords(text)
        
        # Extract structured data
        structured_data = self._extract_structured_data(content.get('html', ''))
        
        processed_content = {
            'title': content.get('title', ''),
            'text': text,
            'url': content.get('url', ''),
            'domain': content.get('domain', ''),
            'ev_keywords': ev_keywords,
            'structured_data': structured_data,
            'content_length': len(text),
            'scraped_at': time.time()
        }
        
        return processed_content
    
    def _extract_ev_keywords(self, text: str) -> List[str]:
        """Extract EV-related keywords from text"""
        ev_keywords = [
            'electric vehicle', 'EV', 'charging station', 'charger', 'battery',
            'plug-in', 'hybrid', 'fast charging', 'level 2', 'level 3',
            'DC fast charging', 'Tesla', 'Supercharger', 'ChargePoint',
            'EVgo', 'Electrify America', 'Blink', 'voltage', 'amperage',
            'kilowatt', 'kWh', 'range', 'miles per charge'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in ev_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_structured_data(self, html: str) -> Dict[str, Any]:
        """Extract structured data from HTML"""
        structured_data = {}
        
        try:
            # Look for JSON-LD structured data
            json_ld_pattern = r'<script type="application/ld\+json">(.*?)</script>'
            json_ld_matches = re.findall(json_ld_pattern, html, re.DOTALL)
            
            for match in json_ld_matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict):
                        structured_data.update(data)
                except json.JSONDecodeError:
                    continue
            
            # Look for microdata
            microdata_pattern = r'itemtype="([^"]*)"'
            microdata_matches = re.findall(microdata_pattern, html)
            if microdata_matches:
                structured_data['microdata_types'] = microdata_matches
            
        except Exception as e:
            self.logger.warning(f"Error extracting structured data: {e}")
        
        return structured_data
    
    async def _find_sub_pages(self, page, domain: str) -> List[str]:
        """Find sub-pages to scrape within the domain"""
        try:
            # Get all links on the page
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href).filter(href => href.startsWith('http'));
                }
            """)
            
            # Filter links to same domain
            domain_links = []
            for link in links:
                if domain in link and link not in self.visited_urls:
                    domain_links.append(link)
                    self.visited_urls.add(link)
            
            return domain_links[:50]  # Limit to 50 sub-pages per domain
            
        except Exception as e:
            self.logger.warning(f"Error finding sub-pages: {e}")
            return []
    
    def save_scraped_data(self, output_path: str = "data/scraped_web_data.json"):
        """Save scraped data to JSON file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Scraped data saved to {output_file}")
            
            # Also save as CSV for analysis
            csv_path = output_path.replace('.json', '.csv')
            df = pd.DataFrame(self.scraped_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"Scraped data also saved as CSV to {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving scraped data: {e}")
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get statistics about the scraping process"""
        total_pages = len(self.scraped_data)
        total_content_length = sum(item.get('content_length', 0) for item in self.scraped_data)
        
        domain_stats = {}
        for item in self.scraped_data:
            domain = item.get('domain', 'unknown')
            if domain not in domain_stats:
                domain_stats[domain] = 0
            domain_stats[domain] += 1
        
        return {
            'total_pages_scraped': total_pages,
            'total_content_length': total_content_length,
            'average_content_length': total_content_length / total_pages if total_pages > 0 else 0,
            'domains_scraped': domain_stats,
            'unique_urls': len(self.visited_urls)
        }
