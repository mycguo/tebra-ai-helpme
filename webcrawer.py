import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebCrawler:
    def __init__(self, url, max_depth):
        self.url = url
        self.subdomains = set()
        self.max_depth = max_depth
        # Extract base domain for filtering
        parsed_url = urlparse(url)
        self.base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.domain_name = parsed_url.netloc

    def start_crawling(self, url):
        return self.crawl(url, depth=0)

    def is_valid_url(self, url):
        """Check if URL belongs to the same domain and is valid"""
        try:
            parsed = urlparse(url)
            return (parsed.netloc == self.domain_name or 
                   parsed.netloc == '' or  # relative URLs
                   parsed.netloc.endswith(f'.{self.domain_name}'))  # subdomains
        except:
            return False
    
    def normalize_url(self, url):
        """Normalize URL by removing fragments, queries, and trailing slashes"""
        try:
            parsed = urlparse(url.strip())
            # Remove fragment (#), query (?), and normalize path
            path = parsed.path if parsed.path else '/'
            
            # Remove trailing slash except for root path
            if path.endswith('/') and path != '/':
                path = path.rstrip('/')
                
            # Rebuild URL without fragment and query
            normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
            return normalized.lower()  # Convert to lowercase for consistency
        except Exception as e:
            print(f"[ERROR] Failed to normalize URL {url}: {e}")
            return url.lower()

    def crawl(self, url, depth):
        # Normalize URL first to check for duplicates
        normalized_url = self.normalize_url(url)
        
        # CRITICAL: Check if URL already processed BEFORE making any requests
        if normalized_url in self.subdomains:
            print(f"[SKIP] Already crawled: {normalized_url}")
            return self.subdomains
        
        # Add to processed set immediately to prevent duplicate crawling
        self.subdomains.add(normalized_url)
        print(f"[INFO] Crawling depth {depth}: {url} (normalized: {normalized_url})")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Encoding": "identity",  # Disable compression to avoid encoding issues
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive"
        }
       
        try:
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Handle encoding explicitly
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as err:
            print(f"[-] An error occurred crawling {url}: {err}")
            return self.subdomains
        except UnicodeDecodeError as err:
            print(f"[-] Encoding error occurred: {err}")
            return self.subdomains

        # Only continue if we haven't reached max depth
        if depth < self.max_depth:
            found_links = set()  # Use set to avoid duplicates within this page
            
            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href or href.strip() == '':
                    continue
                
                # Skip non-HTTP links (mailto:, javascript:, etc.)
                if href.startswith(('mailto:', 'javascript:', 'tel:', 'ftp:')):
                    continue
                
                # Convert relative URLs to absolute
                try:
                    absolute_url = urljoin(url, href.strip())
                except Exception as e:
                    print(f"[WARN] Failed to join URL {href}: {e}")
                    continue
                
                # Normalize the URL
                normalized_link = self.normalize_url(absolute_url)
                
                # Skip if already processed
                if normalized_link in self.subdomains:
                    continue
                    
                # Check if URL belongs to same domain
                if self.is_valid_url(absolute_url):
                    found_links.add(normalized_link)
            
            print(f"[INFO] Found {len(found_links)} new links at depth {depth}")
            
            # Recursively crawl found links
            for link in found_links:
                try:
                    self.crawl(link, depth + 1)
                except Exception as e:
                    print(f"[-] Error crawling {link}: {e}")
                    continue

        return self.subdomains

    def print_results(self):
        print(f"[INFO] Crawling completed. Found {len(self.subdomains)} unique URLs")
        print(f"[INFO] Base domain: {self.domain_name}")
        print(f"[INFO] Max depth: {self.max_depth}")
        print("\n[RESULTS] All URLs discovered:")
        if self.subdomains:
            for i, subdomain in enumerate(sorted(self.subdomains), 1):
                print(f"[{i:03d}]: {subdomain}")
        else:
            print("No URLs found.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', dest='url', help="Specify the URL, provide it along http/https", required=True)
    parser.add_argument('-d', '--depth', dest='depth', type=int, default=1, help="Specify the recursion depth limit")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    web_crawler = WebCrawler(args.url, args.depth)
    web_crawler.start_crawling(args.url)
    web_crawler.print_results()