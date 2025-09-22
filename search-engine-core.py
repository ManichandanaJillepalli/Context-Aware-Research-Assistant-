"""
Core Search Engine for Academic Papers
Integrates multiple academic APIs with bias detection and ranking
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config import Config
from ..utils.ethics_checker import EthicsChecker
from ..models.embedding_model import EmbeddingModel

@dataclass
class SearchResult:
    """Represents a single search result from academic sources"""
    title: str
    abstract: str
    authors: List[str]
    publication_date: datetime
    source: str
    url: str
    doi: Optional[str] = None
    citation_count: int = 0
    venue: Optional[str] = None
    pdf_url: Optional[str] = None
    keywords: List[str] = None
    relevance_score: float = 0.0
    bias_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "publication_date": self.publication_date.isoformat(),
            "source": self.source,
            "url": self.url,
            "doi": self.doi,
            "citation_count": self.citation_count,
            "venue": self.venue,
            "pdf_url": self.pdf_url,
            "keywords": self.keywords or [],
            "relevance_score": self.relevance_score,
            "bias_score": self.bias_score
        }

class AcademicSearchEngine:
    """
    Advanced academic search engine that prioritizes scholarly sources
    and implements bias detection and mitigation strategies.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ethics_checker = EthicsChecker()
        self.embedding_model = EmbeddingModel()
        
        # API clients
        self.semantic_scholar_session = None
        self.arxiv_session = None
        
        # Search parameters
        self.max_results_per_source = 50
        self.bias_threshold = 0.7
        self.min_citation_count = 5
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.semantic_scholar_session = aiohttp.ClientSession()
        self.arxiv_session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.semantic_scholar_session:
            await self.semantic_scholar_session.close()
        if self.arxiv_session:
            await self.arxiv_session.close()
    
    async def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 100,
        time_range: Tuple[datetime, datetime] = None,
        bias_filter: bool = True,
        citation_threshold: int = 0,
        fields_of_study: List[str] = None
    ) -> List[SearchResult]:
        """
        Comprehensive academic search across multiple sources
        
        Args:
            query: Search query string
            sources: List of sources to search ['semantic_scholar', 'arxiv']
            max_results: Maximum number of results to return
            time_range: Tuple of (start_date, end_date) for filtering
            bias_filter: Whether to apply bias detection and filtering
            citation_threshold: Minimum citation count for results
            fields_of_study: Filter by specific academic fields
            
        Returns:
            List of SearchResult objects ranked by relevance and quality
        """
        if sources is None:
            sources = ['semantic_scholar', 'arxiv']
        
        self.logger.info(f"Starting search for query: '{query}' across sources: {sources}")
        
        # Check ethical compliance
        if not self.ethics_checker.is_query_ethical(query):
            raise ValueError(f"Query violates ethical guidelines: {query}")
        
        # Parallel search across all sources
        search_tasks = []
        if 'semantic_scholar' in sources:
            search_tasks.append(self._search_semantic_scholar(
                query, max_results // len(sources), time_range, fields_of_study
            ))
        if 'arxiv' in sources:
            search_tasks.append(self._search_arxiv(
                query, max_results // len(sources), time_range, fields_of_study
            ))
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine and deduplicate results
        all_results = []
        for source_results in search_results:
            all_results.extend(source_results)
        
        deduplicated_results = self._deduplicate_results(all_results)
        
        # Apply filters
        filtered_results = self._apply_filters(
            deduplicated_results, 
            citation_threshold, 
            bias_filter
        )
        
        # Rank and score results
        ranked_results = await self._rank_results(filtered_results, query)
        
        # Return top results
        return ranked_results[:max_results]
    
    async def _search_semantic_scholar(
        self,
        query: str,
        max_results: int,
        time_range: Tuple[datetime, datetime] = None,
        fields_of_study: List[str] = None
    ) -> List[SearchResult]:
        """Search Semantic Scholar API"""
        if not self.semantic_scholar_session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API limit
            "fields": "paperId,title,abstract,authors,year,citationCount,venue,url,openAccessPdf,fieldsOfStudy"
        }
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        if time_range:
            start_year = time_range[0].year
            end_year = time_range[1].year
            params["year"] = f"{start_year}-{end_year}"
        
        try:
            # Rate limiting compliance
            await self.ethics_checker.check_rate_limit("semantic_scholar")
            
            async with self.semantic_scholar_session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_semantic_scholar_results(data.get("data", []))
                else:
                    self.logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    async def _search_arxiv(
        self,
        query: str,
        max_results: int,
        time_range: Tuple[datetime, datetime] = None,
        fields_of_study: List[str] = None
    ) -> List[SearchResult]:
        """Search ArXiv API"""
        if not self.arxiv_session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        base_url = "http://export.arxiv.org/api/query"
        
        # Build search query
        search_query = f"all:{query}"
        if fields_of_study:
            # Map fields to ArXiv categories
            arxiv_categories = self._map_fields_to_arxiv_categories(fields_of_study)
            if arxiv_categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in arxiv_categories])
                search_query = f"({search_query}) AND ({cat_query})"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 1000),  # ArXiv allows up to 1000
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            # Rate limiting compliance
            await self.ethics_checker.check_rate_limit("arxiv")
            
            async with self.arxiv_session.get(base_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_arxiv_results(content, time_range)
                else:
                    self.logger.error(f"ArXiv API error: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _parse_semantic_scholar_results(self, data: List[Dict]) -> List[SearchResult]:
        """Parse Semantic Scholar API response"""
        results = []
        for paper in data:
            try:
                # Extract publication date
                pub_date = datetime(paper.get("year", 2000), 1, 1) if paper.get("year") else datetime.now()
                
                # Extract authors
                authors = [author.get("name", "Unknown") for author in paper.get("authors", [])]
                
                # Extract PDF URL
                pdf_url = None
                if paper.get("openAccessPdf"):
                    pdf_url = paper["openAccessPdf"].get("url")
                
                result = SearchResult(
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    authors=authors,
                    publication_date=pub_date,
                    source="semantic_scholar",
                    url=paper.get("url", ""),
                    doi=None,  # Not provided in basic search
                    citation_count=paper.get("citationCount", 0),
                    venue=paper.get("venue", ""),
                    pdf_url=pdf_url,
                    keywords=paper.get("fieldsOfStudy", [])
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error parsing Semantic Scholar result: {e}")
                continue
        
        return results
    
    def _parse_arxiv_results(self, xml_content: str, time_range: Tuple[datetime, datetime] = None) -> List[SearchResult]:
        """Parse ArXiv API XML response"""
        import xml.etree.ElementTree as ET
        
        results = []
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', namespaces):
                try:
                    # Extract basic information
                    title = entry.find('atom:title', namespaces).text.strip()
                    abstract = entry.find('atom:summary', namespaces).text.strip()
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name = author.find('atom:name', namespaces)
                        if name is not None:
                            authors.append(name.text)
                    
                    # Extract publication date
                    published = entry.find('atom:published', namespaces).text
                    pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    
                    # Apply time filter
                    if time_range and (pub_date < time_range[0] or pub_date > time_range[1]):
                        continue
                    
                    # Extract URLs
                    url = entry.find('atom:id', namespaces).text
                    pdf_url = None
                    for link in entry.findall('atom:link', namespaces):
                        if link.get('type') == 'application/pdf':
                            pdf_url = link.get('href')
                            break
                    
                    # Extract categories (keywords)
                    categories = []
                    for category in entry.findall('atom:category', namespaces):
                        categories.append(category.get('term'))
                    
                    result = SearchResult(
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        publication_date=pub_date,
                        source="arxiv",
                        url=url,
                        doi=None,
                        citation_count=0,  # ArXiv doesn't provide citation counts
                        venue="arXiv preprint",
                        pdf_url=pdf_url,
                        keywords=categories
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error parsing ArXiv entry: {e}")
                    continue
        
        except ET.ParseError as e:
            self.logger.error(f"Error parsing ArXiv XML: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on title similarity"""
        if not results:
            return []
        
        # Create embeddings for all titles
        titles = [result.title for result in results]
        embeddings = self.embedding_model.encode(titles)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find duplicates
        to_remove = set()
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] > 0.9:  # High similarity threshold
                    # Keep the one with higher citation count
                    if results[i].citation_count >= results[j].citation_count:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
        
        # Return deduplicated results
        return [result for i, result in enumerate(results) if i not in to_remove]
    
    def _apply_filters(
        self, 
        results: List[SearchResult], 
        citation_threshold: int, 
        bias_filter: bool
    ) -> List[SearchResult]:
        """Apply various filters to search results"""
        filtered_results = []
        
        for result in results:
            # Citation threshold filter
            if result.citation_count < citation_threshold:
                continue
            
            # Bias detection and filtering
            if bias_filter:
                bias_score = self._calculate_bias_score(result)
                result.bias_score = bias_score
                if bias_score > self.bias_threshold:
                    continue
            
            # Quality filters
            if not result.title or len(result.title) < 10:
                continue
            if not result.abstract or len(result.abstract) < 100:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_bias_score(self, result: SearchResult) -> float:
        """Calculate bias score for a research result"""
        bias_indicators = [
            # Check for sensational language in title
            any(word in result.title.lower() for word in [
                'revolutionary', 'breakthrough', 'game-changing', 'unprecedented'
            ]),
            
            # Check for single author papers (higher bias risk)
            len(result.authors) == 1,
            
            # Check for very recent papers without peer review
            (datetime.now() - result.publication_date).days < 30 and result.source == 'arxiv',
            
            # Check for very low or very high citation counts (outliers)
            result.citation_count > 0 and (
                result.citation_count < 2 or 
                result.citation_count > 10000
            )
        ]
        
        # Calculate weighted bias score
        bias_score = sum(bias_indicators) / len(bias_indicators)
        return bias_score
    
    async def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank results by relevance and quality metrics"""
        if not results:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate relevance scores
        for result in results:
            # Semantic similarity score
            text_for_embedding = f"{result.title} {result.abstract}"
            result_embedding = self.embedding_model.encode([text_for_embedding])[0]
            semantic_score = cosine_similarity([query_embedding], [result_embedding])[0][0]
            
            # Quality score based on citations, venue, etc.
            quality_score = self._calculate_quality_score(result)
            
            # Recency score (recent papers get slight boost)
            days_old = (datetime.now() - result.publication_date).days
            recency_score = max(0, 1 - days_old / 365)  # Decay over year
            
            # Combined relevance score
            result.relevance_score = (
                0.6 * semantic_score + 
                0.3 * quality_score + 
                0.1 * recency_score
            )
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _calculate_quality_score(self, result: SearchResult) -> float:
        """Calculate quality score based on various factors"""
        score = 0.0
        
        # Citation score (normalized)
        if result.citation_count > 0:
            # Log transform to handle very high citation counts
            citation_score = min(1.0, np.log(result.citation_count + 1) / np.log(1000))
            score += 0.4 * citation_score
        
        # Venue score
        if result.venue:
            # High-impact venue detection
            high_impact_venues = [
                'nature', 'science', 'cell', 'pnas', 'nejm', 'ieee', 'acm', 'neurips'
            ]
            if any(venue in result.venue.lower() for venue in high_impact_venues):
                score += 0.3
            else:
                score += 0.1  # Other venues get partial credit
        
        # Author count (collaborative work often higher quality)
        if len(result.authors) > 1:
            score += min(0.2, 0.05 * len(result.authors))
        
        # Abstract length (detailed abstracts indicate thorough work)
        if result.abstract:
            abstract_score = min(0.1, len(result.abstract) / 2000)
            score += abstract_score
        
        return min(1.0, score)
    
    def _map_fields_to_arxiv_categories(self, fields: List[str]) -> List[str]:
        """Map academic fields to ArXiv category codes"""
        field_mapping = {
            'computer science': ['cs'],
            'machine learning': ['cs.LG', 'stat.ML'],
            'artificial intelligence': ['cs.AI'],
            'physics': ['physics'],
            'mathematics': ['math'],
            'biology': ['q-bio'],
            'neuroscience': ['q-bio.NC'],
            'statistics': ['stat'],
            'economics': ['econ']
        }
        
        categories = []
        for field in fields:
            field_lower = field.lower()
            for key, cats in field_mapping.items():
                if key in field_lower:
                    categories.extend(cats)
        
        return list(set(categories))  # Remove duplicates

# Example usage and testing
async def main():
    """Example usage of the Academic Search Engine"""
    config = Config()
    
    async with AcademicSearchEngine(config) as search_engine:
        results = await search_engine.search(
            query="neuromorphic computing applications",
            sources=["semantic_scholar", "arxiv"],
            max_results=20,
            bias_filter=True,
            citation_threshold=5,
            fields_of_study=["computer science", "neuroscience"]
        )
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result.title}")
            print(f"   Authors: {', '.join(result.authors[:3])}...")
            print(f"   Citations: {result.citation_count}")
            print(f"   Relevance: {result.relevance_score:.3f}")
            print(f"   Bias Score: {result.bias_score:.3f}")

if __name__ == "__main__":
    asyncio.run(main())