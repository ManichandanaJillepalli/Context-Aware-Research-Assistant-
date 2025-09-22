"""
Citation Management System
Handles automatic citation generation, verification, and formatting
"""
import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import requests
from urllib.parse import urlparse
import bibtexparser
from habanero import Crossref
import scholarly

from .search_engine import SearchResult
from ..utils.config import Config

@dataclass
class Citation:
    """Represents a citation with multiple format options"""
    title: str
    authors: List[str]
    publication_date: datetime
    venue: str
    url: str
    doi: Optional[str] = None
    pages: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    
    # Formatted citations
    apa_citation: str = ""
    mla_citation: str = ""
    chicago_citation: str = ""
    ieee_citation: str = ""
    bibtex: str = ""
    
    # Quality metrics
    link_verified: bool = False
    accessible: bool = False
    citation_accuracy: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "venue": self.venue,
            "url": self.url,
            "doi": self.doi,
            "pages": self.pages,
            "volume": self.volume,
            "issue": self.issue,
            "publisher": self.publisher,
            "abstract": self.abstract,
            "apa_citation": self.apa_citation,
            "mla_citation": self.mla_citation,
            "chicago_citation": self.chicago_citation,
            "ieee_citation": self.ieee_citation,
            "bibtex": self.bibtex,
            "link_verified": self.link_verified,
            "accessible": self.accessible,
            "citation_accuracy": self.citation_accuracy
        }
    
    @property
    def formatted_citation(self) -> str:
        """Return the default formatted citation (APA)"""
        return self.apa_citation

class CitationManager:
    """
    Advanced citation management system with automatic generation,
    verification, and multiple format support
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize external services
        self.crossref_client = Crossref()
        self.session = None
        
        # Citation formatting templates
        self._init_citation_templates()
        
        # Quality thresholds
        self.link_timeout = 10  # seconds
        self.accuracy_threshold = 0.8
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.link_timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _init_citation_templates(self):
        """Initialize citation formatting templates"""
        self.templates = {
            'apa': {
                'article': "{authors} ({year}). {title}. {venue}, {volume}({issue}), {pages}. {doi_url}",
                'preprint': "{authors} ({year}). {title}. {venue}. {url}",
                'book': "{authors} ({year}). {title}. {publisher}.",
                'chapter': "{authors} ({year}). {title}. In {editors} (Eds.), {book_title} (pp. {pages}). {publisher}."
            },
            'mla': {
                'article': "{author_last}, {author_first}. \"{title}.\" {venue}, vol. {volume}, no. {issue}, {year}, pp. {pages}. {doi_url}",
                'preprint': "{author_last}, {author_first}. \"{title}.\" {venue}, {year}, {url}.",
                'book': "{author_last}, {author_first}. {title}. {publisher}, {year}.",
                'chapter': "{author_last}, {author_first}. \"{title}.\" {book_title}, edited by {editors}, {publisher}, {year}, pp. {pages}."
            },
            'chicago': {
                'article': "{authors}. \"{title}.\" {venue} {volume}, no. {issue} ({year}): {pages}. {doi_url}",
                'preprint': "{authors}. \"{title}.\" {venue}, {year}. {url}.",
                'book': "{authors}. {title}. {publisher}, {year}.",
                'chapter': "{authors}. \"{title}.\" In {book_title}, edited by {editors}, {pages}. {publisher}, {year}."
            },
            'ieee': {
                'article': "{authors}, \"{title},\" {venue}, vol. {volume}, no. {issue}, pp. {pages}, {year}. {doi_url}",
                'preprint': "{authors}, \"{title},\" {venue}, {year}. Available: {url}",
                'book': "{authors}, {title}. {publisher}, {year}.",
                'chapter': "{authors}, \"{title},\" in {book_title}, {editors}, Eds. {publisher}, {year}, pp. {pages}."
            }
        }
    
    async def generate_citations(
        self,
        sources: List[SearchResult],
        citation_style: str = "apa",
        verify_links: bool = True,
        enrich_metadata: bool = True
    ) -> List[Citation]:
        """
        Generate citations from search results
        
        Args:
            sources: List of search results to cite
            citation_style: Primary citation style ('apa', 'mla', 'chicago', 'ieee')
            verify_links: Whether to verify link accessibility
            enrich_metadata: Whether to enrich metadata from external sources
            
        Returns:
            List of Citation objects with formatted citations
        """
        self.logger.info(f"Generating citations for {len(sources)} sources")
        
        citations = []
        for source in sources:
            try:
                citation = await self._create_citation_from_source(
                    source, enrich_metadata
                )
                
                # Generate all citation formats
                await self._format_all_citations(citation)
                
                # Verify link accessibility if requested
                if verify_links and citation.url:
                    citation.link_verified, citation.accessible = await self._verify_link(citation.url)
                
                # Calculate citation accuracy
                citation.citation_accuracy = self._calculate_citation_accuracy(citation)
                
                citations.append(citation)
                
            except Exception as e:
                self.logger.error(f"Error generating citation for {source.title}: {e}")
                continue
        
        return citations
    
    async def _create_citation_from_source(
        self, 
        source: SearchResult, 
        enrich_metadata: bool = True
    ) -> Citation:
        """Create a Citation object from a SearchResult"""
        citation = Citation(
            title=source.title,
            authors=source.authors,
            publication_date=source.publication_date,
            venue=source.venue or "Unknown",
            url=source.url,
            doi=source.doi,
            abstract=source.abstract
        )
        
        # Enrich metadata from external sources
        if enrich_metadata:
            await self._enrich_citation_metadata(citation)
        
        return citation
    
    async def _enrich_citation_metadata(self, citation: Citation):
        """Enrich citation metadata from external APIs"""
        try:
            # Try to find DOI if not present
            if not citation.doi and citation.title:
                citation.doi = await self._find_doi_by_title(citation.title)
            
            # Enrich from Crossref if DOI is available
            if citation.doi:
                await self._enrich_from_crossref(citation)
            
            # Try to enrich from Google Scholar
            await self._enrich_from_google_scholar(citation)
            
        except Exception as e:
            self.logger.warning(f"Could not enrich metadata for {citation.title}: {e}")
    
    async def _find_doi_by_title(self, title: str) -> Optional[str]:
        """Find DOI using Crossref search"""
        try:
            results = self.crossref_client.works(query_title=title, limit=1)
            if results['message']['items']:
                return results['message']['items'][0].get('DOI')
        except Exception:
            pass
        return None
    
    async def _enrich_from_crossref(self, citation: Citation):
        """Enrich citation from Crossref data"""
        try:
            if not citation.doi:
                return
            
            result = self.crossref_client.works(ids=citation.doi)
            work = result['message']
            
            # Update citation information
            if 'container-title' in work:
                citation.venue = work['container-title'][0]
            
            if 'volume' in work:
                citation.volume = str(work['volume'])
            
            if 'issue' in work:
                citation.issue = str(work['issue'])
            
            if 'page' in work:
                citation.pages = work['page']
            
            if 'publisher' in work:
                citation.publisher = work['publisher']
            
            # Update publication date if available
            if 'published-print' in work:
                date_parts = work['published-print']['date-parts'][0]
                if len(date_parts) >= 3:
                    citation.publication_date = datetime(date_parts[0], date_parts[1], date_parts[2])
                elif len(date_parts) >= 1:
                    citation.publication_date = datetime(date_parts[0], 1, 1)
            
            # Update authors if available
            if 'author' in work:
                authors = []
                for author in work['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if family:
                        authors.append(f"{family}, {given}".strip(', '))
                if authors:
                    citation.authors = authors
                    
        except Exception as e:
            self.logger.warning(f"Could not enrich from Crossref: {e}")
    
    async def _enrich_from_google_scholar(self, citation: Citation):
        """Enrich citation from Google Scholar (limited due to rate limits)"""
        try:
            # This is a simplified version due to Google Scholar's strict rate limiting
            # In production, you'd want to implement proper rate limiting and caching
            search_query = scholarly.search_pubs(citation.title)
            pub = next(search_query, None)
            
            if pub:
                filled_pub = scholarly.fill(pub)
                
                # Update venue if not available
                if not citation.venue and 'venue' in filled_pub['bib']:
                    citation.venue = filled_pub['bib']['venue']
                
                # Update publication year
                if 'pub_year' in filled_pub['bib']:
                    year = int(filled_pub['bib']['pub_year'])
                    citation.publication_date = datetime(year, 1, 1)
                
        except Exception as e:
            self.logger.warning(f"Could not enrich from Google Scholar: {e}")
    
    async def _format_all_citations(self, citation: Citation):
        """Generate all citation formats for a citation"""
        # Determine publication type
        pub_type = self._determine_publication_type(citation)
        
        # Generate each format
        citation.apa_citation = self._format_citation(citation, 'apa', pub_type)
        citation.mla_citation = self._format_citation(citation, 'mla', pub_type)
        citation.chicago_citation = self._format_citation(citation, 'chicago', pub_type)
        citation.ieee_citation = self._format_citation(citation, 'ieee', pub_type)
        citation.bibtex = self._generate_bibtex(citation, pub_type)
    
    def _determine_publication_type(self, citation: Citation) -> str:
        """Determine the type of publication for citation formatting"""
        venue_lower = citation.venue.lower() if citation.venue else ""
        
        if any(keyword in venue_lower for keyword in ['arxiv', 'preprint', 'bioRxiv']):
            return 'preprint'
        elif any(keyword in venue_lower for keyword in ['journal', 'proceedings', 'conference']):
            return 'article'
        elif any(keyword in venue_lower for keyword in ['book', 'chapter']):
            return 'chapter' if 'chapter' in venue_lower else 'book'
        else:
            return 'article'  # Default to article
    
    def _format_citation(self, citation: Citation, style: str, pub_type: str) -> str:
        """Format a citation according to the specified style"""
        template = self.templates[style].get(pub_type, self.templates[style]['article'])
        
        # Prepare author formatting
        authors_formatted = self._format_authors(citation.authors, style)
        
        # Prepare date
        year = citation.publication_date.year if citation.publication_date else "n.d."
        
        # Prepare DOI URL
        doi_url = f"https://doi.org/{citation.doi}" if citation.doi else ""
        if not doi_url and citation.url:
            doi_url = citation.url
        
        # Format the citation
        formatted = template.format(
            authors=authors_formatted,
            author_last=citation.authors[0].split(',')[0] if citation.authors else "Unknown",
            author_first=citation.authors[0].split(',')[1].strip() if citation.authors and ',' in citation.authors[0] else "",
            year=year,
            title=citation.title,
            venue=citation.venue or "Unknown Venue",
            volume=citation.volume or "",
            issue=citation.issue or "",
            pages=citation.pages or "",
            publisher=citation.publisher or "",
            doi_url=doi_url,
            url=citation.url or ""
        )
        
        # Clean up extra spaces and formatting
        formatted = re.sub(r'\s+', ' ', formatted)
        formatted = re.sub(r'\s*,\s*,', ',', formatted)  # Remove double commas
        formatted = re.sub(r'\s*\.\s*\.', '.', formatted)  # Remove double periods
        formatted = formatted.strip()
        
        return formatted
    
    def _format_authors(self, authors: List[str], style: str) -> str:
        """Format author list according to citation style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            if style == 'apa':
                return f"{authors[0]} & {authors[1]}"
            else:
                return f"{authors[0]} and {authors[1]}"
        else:
            if style == 'apa':
                return f"{authors[0]}, et al."
            elif style in ['mla', 'chicago']:
                return f"{authors[0]}, et al."
            else:  # IEEE
                return f"{authors[0]} et al."
    
    def _generate_bibtex(self, citation: Citation, pub_type: str) -> str:
        """Generate BibTeX entry"""
        # Create a key from first author's last name and year
        key = self._create_bibtex_key(citation)
        
        # Determine BibTeX entry type
        bibtex_type = {
            'article': 'article',
            'preprint': 'misc',
            'book': 'book',
            'chapter': 'incollection'
        }.get(pub_type, 'article')
        
        # Build BibTeX entry
        bibtex_lines = [f"@{bibtex_type}{{{key},"]
        
        # Add required fields
        bibtex_lines.append(f"  title = {{{citation.title}}},")
        
        if citation.authors:
            authors_str = " and ".join(citation.authors)
            bibtex_lines.append(f"  author = {{{authors_str}}},")
        
        if citation.publication_date:
            bibtex_lines.append(f"  year = {{{citation.publication_date.year}}},")
        
        if citation.venue:
            journal_field = 'journal' if pub_type == 'article' else 'howpublished'
            bibtex_lines.append(f"  {journal_field} = {{{citation.venue}}},")
        
        if citation.volume:
            bibtex_lines.append(f"  volume = {{{citation.volume}}},")
        
        if citation.issue:
            bibtex_lines.append(f"  number = {{{citation.issue}}},")
        
        if citation.pages:
            bibtex_lines.append(f"  pages = {{{citation.pages}}},")
        
        if citation.doi:
            bibtex_lines.append(f"  doi = {{{citation.doi}}},")
        
        if citation.url:
            bibtex_lines.append(f"  url = {{{citation.url}}},")
        
        # Remove trailing comma from last line
        if bibtex_lines[-1].endswith(','):
            bibtex_lines[-1] = bibtex_lines[-1][:-1]
        
        bibtex_lines.append("}")
        
        return "\n".join(bibtex_lines)
    
    def _create_bibtex_key(self, citation: Citation) -> str:
        """Create a unique BibTeX key"""
        # Use first author's last name and year
        if citation.authors:
            first_author = citation.authors[0].split(',')[0].lower()
            # Remove special characters
            first_author = re.sub(r'[^a-z0-9]', '', first_author)
        else:
            first_author = "unknown"
        
        year = citation.publication_date.year if citation.publication_date else "nodate"
        
        return f"{first_author}{year}"
    
    async def _verify_link(self, url: str) -> Tuple[bool, bool]:
        """Verify if a link is valid and accessible"""
        if not self.session:
            return False, False
        
        try:
            async with self.session.head(url) as response:
                link_verified = True
                accessible = 200 <= response.status < 400
                return link_verified, accessible
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout verifying link: {url}")
            return True, False  # Link exists but slow
        except Exception as e:
            self.logger.warning(f"Error verifying link {url}: {e}")
            return False, False
    
    def _calculate_citation_accuracy(self, citation: Citation) -> float:
        """Calculate accuracy score for a citation"""
        score = 0.0
        total_fields = 0
        
        # Check required fields
        required_fields = [
            ('title', citation.title),
            ('authors', citation.authors),
            ('publication_date', citation.publication_date),
            ('venue', citation.venue)
        ]
        
        for field_name, field_value in required_fields:
            total_fields += 1
            if field_value:
                if field_name == 'authors' and len(field_value) > 0:
                    score += 1
                elif field_name != 'authors':
                    score += 1
        
        # Bonus for additional fields
        optional_fields = [
            citation.doi,
            citation.volume,
            citation.issue,
            citation.pages,
            citation.publisher
        ]
        
        for field in optional_fields:
            total_fields += 0.2  # Less weight for optional fields
            if field:
                score += 0.2
        
        # Bonus for verified links
        if citation.link_verified:
            score += 0.1
            total_fields += 0.1
        
        return min(1.0, score / total_fields) if total_fields > 0 else 0.0
    
    async def generate_bibliography(
        self,
        citations: List[Citation],
        style: str = "apa",
        sort_alphabetically: bool = True
    ) -> str:
        """Generate a formatted bibliography"""
        if not citations:
            return ""
        
        # Get formatted citations
        formatted_citations = []
        for citation in citations:
            if style == 'apa':
                formatted = citation.apa_citation
            elif style == 'mla':
                formatted = citation.mla_citation
            elif style == 'chicago':
                formatted = citation.chicago_citation
            elif style == 'ieee':
                formatted = citation.ieee_citation
            else:
                formatted = citation.apa_citation
            
            formatted_citations.append(formatted)
        
        # Sort alphabetically if requested
        if sort_alphabetically:
            formatted_citations.sort()
        
        # Create bibliography
        bibliography = "\n\n".join(formatted_citations)
        
        return bibliography
    
    async def validate_citations(self, citations: List[Citation]) -> Dict[str, any]:
        """Validate a list of citations and return quality metrics"""
        if not citations:
            return {"total_citations": 0}
        
        metrics = {
            "total_citations": len(citations),
            "verified_links": sum(1 for c in citations if c.link_verified),
            "accessible_links": sum(1 for c in citations if c.accessible),
            "with_doi": sum(1 for c in citations if c.doi),
            "average_accuracy": sum(c.citation_accuracy for c in citations) / len(citations),
            "high_quality": sum(1 for c in citations if c.citation_accuracy > self.accuracy_threshold),
            "complete_metadata": sum(1 for c in citations if all([
                c.title, c.authors, c.publication_date, c.venue
            ]))
        }
        
        # Calculate percentages
        total = metrics["total_citations"]
        metrics.update({
            "link_verification_rate": metrics["verified_links"] / total * 100,
            "accessibility_rate": metrics["accessible_links"] / total * 100,
            "doi_coverage": metrics["with_doi"] / total * 100,
            "high_quality_rate": metrics["high_quality"] / total * 100,
            "completeness_rate": metrics["complete_metadata"] / total * 100
        })
        
        return metrics

# Example usage
async def main():
    """Example usage of the Citation Manager"""
    from .search_engine import SearchResult
    from ..utils.config import Config
    
    config = Config()
    
    # Mock search result
    mock_result = SearchResult(
        title="Advances in Neural Network Architecture Search",
        abstract="This paper presents novel approaches to neural architecture search...",
        authors=["Smith, John A.", "Doe, Jane B."],
        publication_date=datetime(2023, 3, 15),
        source="semantic_scholar",
        url="https://example.com/paper",
        doi="10.1000/example.doi",
        citation_count=42,
        venue="Journal of Machine Learning Research"
    )
    
    async with CitationManager(config) as citation_manager:
        citations = await citation_manager.generate_citations(
            sources=[mock_result],
            citation_style="apa",
            verify_links=True,
            enrich_metadata=True
        )
        
        print("Generated Citations:")
        for citation in citations:
            print(f"\nAPA: {citation.apa_citation}")
            print(f"MLA: {citation.mla_citation}")
            print(f"Accuracy: {citation.citation_accuracy:.3f}")
            print(f"Link verified: {citation.link_verified}")
        
        # Generate bibliography
        bibliography = await citation_manager.generate_bibliography(
            citations, style="apa"
        )
        print(f"\nBibliography:\n{bibliography}")
        
        # Validate citations
        validation = await citation_manager.validate_citations(citations)
        print(f"\nValidation metrics: {validation}")

if __name__ == "__main__":
    asyncio.run(main())