import arxiv
import os
import sys
import ssl
import certifi

# Add project root to sys.path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import Config

class ArxivIngestor:
    def __init__(self, max_results=Config.MAX_PAPERS):
        self.max_results = max_results
        self.download_dir = Config.PAPERS_DIR
        
        # SSL Fix
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        # Fallback if certifi doesn't help in some envs
        if not os.environ.get("VERIFY_SSL", "True") == "True":
             self.ssl_context = ssl._create_unverified_context()

    def _clean_download_dir(self):
        """Removes all files in the download directory."""
        if os.path.exists(self.download_dir):
            for file in os.listdir(self.download_dir):
                file_path = os.path.join(self.download_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            os.makedirs(self.download_dir, exist_ok=True)

    def search_and_download(self, query: str):
        """
        Searches arXiv for papers and downloads them.
        Clears existing papers first.
        """
        self._clean_download_dir()
        
        print(f"Searching arXiv for: {query}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        downloaded_papers = []
        
        # Create a list from results to handle generator safely
        try:
            results = list(client.results(search))
        except Exception as e:
            print(f"Error fetching results: {e}")
            return []
            
        for result in results:
            try:
                # Clean filename
                safe_title = "".join([c for c in result.title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
                filename = f"{safe_title}.pdf"
                filepath = os.path.join(self.download_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Downloading: {result.title}")
                    # Custom download to handle SSL and User-Agent
                    import urllib.request
                    req = urllib.request.Request(
                        result.pdf_url, 
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req, context=self.ssl_context) as response, open(filepath, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                        
                    # Verify download
                    if os.path.getsize(filepath) < 1000:
                        print(f"Warning: Downloaded file is too small ({os.path.getsize(filepath)} bytes). Deleting.")
                        os.remove(filepath)
                else:
                    print(f"Already exists: {result.title}")
                    
                downloaded_papers.append({
                    "title": result.title,
                    "filepath": filepath,
                    "summary": result.summary,
                    "authors": [a.name for a in result.authors],
                    "published": str(result.published),
                    "url": result.pdf_url
                })
            except Exception as e:
                print(f"Failed to download {result.title}: {e}")
                
        return downloaded_papers

if __name__ == "__main__":
    ingestor = ArxivIngestor(max_results=3)
    papers = ingestor.search_and_download("RAG Retrieval Augmented Generation")
    print(f"Downloaded {len(papers)} papers.")
