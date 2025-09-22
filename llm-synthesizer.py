"""
LLM-based Synthesis and Summarization Module
Handles multi-document synthesis with citation-aware generation
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import openai
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, BartTokenizer, BartForConditionalGeneration
)
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_engine import SearchResult
from ..utils.config import Config
from .citation_manager import Citation, CitationManager

@dataclass
class SynthesisResult:
    """Result of multi-document synthesis"""
    synthesized_text: str
    sources_used: List[SearchResult]
    citations: List[Citation]
    confidence_score: float
    synthesis_type: str
    word_count: int
    key_findings: List[str]
    gaps_identified: List[str]
    contradictions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "synthesized_text": self.synthesized_text,
            "sources_used": [source.to_dict() for source in self.sources_used],
            "citations": [citation.to_dict() for citation in self.citations],
            "confidence_score": self.confidence_score,
            "synthesis_type": self.synthesis_type,
            "word_count": self.word_count,
            "key_findings": self.key_findings,
            "gaps_identified": self.gaps_identified,
            "contradictions": self.contradictions
        }

class LLMSynthesizer:
    """
    Advanced LLM-based synthesizer for academic content
    Supports multiple synthesis strategies and citation generation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.citation_manager = CitationManager(config)
        
        # Initialize models
        self._init_models()
        
        # Synthesis parameters
        self.max_input_length = 4000  # Tokens
        self.max_output_length = 1500  # Tokens
        self.chunk_overlap = 200  # Tokens
        self.min_confidence_threshold = 0.7
        
    def _init_models(self):
        """Initialize LLM and supporting models"""
        # Main synthesis model (BART fine-tuned for academic content)
        self.synthesis_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.synthesis_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        # Sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        # OpenAI client for advanced synthesis
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        
        # Claim verification model
        self.claim_verifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            return_all_scores=True
        )
    
    async def synthesize(
        self,
        documents: List[SearchResult],
        query: str,
        synthesis_type: str = "comprehensive_review",
        citation_style: str = "apa",
        max_length: int = 1500,
        include_contradictions: bool = True
    ) -> SynthesisResult:
        """
        Synthesize information from multiple documents
        
        Args:
            documents: List of search results to synthesize
            query: Original research query
            synthesis_type: Type of synthesis ('comprehensive_review', 'comparison', 'summary')
            citation_style: Citation format ('apa', 'mla', 'chicago')
            max_length: Maximum length of synthesized text
            include_contradictions: Whether to identify and include contradictions
            
        Returns:
            SynthesisResult with synthesized content and metadata
        """
        self.logger.info(f"Starting synthesis for {len(documents)} documents")
        
        if not documents:
            raise ValueError("No documents provided for synthesis")
        
        # Preprocess documents
        processed_docs = await self._preprocess_documents(documents, query)
        
        # Group documents by similarity for better synthesis
        doc_groups = self._group_similar_documents(processed_docs)
        
        # Generate synthesis based on type
        if synthesis_type == "comprehensive_review":
            synthesis_text = await self._generate_comprehensive_review(
                doc_groups, query, max_length
            )
        elif synthesis_type == "comparison":
            synthesis_text = await self._generate_comparison(
                doc_groups, query, max_length
            )
        elif synthesis_type == "summary":
            synthesis_text = await self._generate_summary(
                doc_groups, query, max_length
            )
        else:
            raise ValueError(f"Unknown synthesis type: {synthesis_type}")
        
        # Generate citations
        citations = await self.citation_manager.generate_citations(
            processed_docs, citation_style
        )
        
        # Insert citations into text
        cited_text = await self._insert_citations(synthesis_text, citations)
        
        # Identify key findings, gaps, and contradictions
        key_findings = await self._extract_key_findings(cited_text)
        gaps_identified = await self._identify_knowledge_gaps(processed_docs, query)
        contradictions = []
        if include_contradictions:
            contradictions = await self._find_contradictions(processed_docs)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            processed_docs, synthesis_text
        )
        
        return SynthesisResult(
            synthesized_text=cited_text,
            sources_used=documents,
            citations=citations,
            confidence_score=confidence_score,
            synthesis_type=synthesis_type,
            word_count=len(cited_text.split()),
            key_findings=key_findings,
            gaps_identified=gaps_identified,
            contradictions=contradictions
        )
    
    async def _preprocess_documents(
        self, 
        documents: List[SearchResult], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Preprocess documents for synthesis"""
        processed_docs = []
        
        for doc in documents:
            # Combine title and abstract for processing
            full_text = f"{doc.title}\n\n{doc.abstract}"
            
            # Chunk long documents
            chunks = self._chunk_text(full_text, self.max_input_length)
            
            # Score relevance of each chunk to query
            chunk_scores = await self._score_chunk_relevance(chunks, query)
            
            # Keep most relevant chunks
            top_chunks = sorted(
                zip(chunks, chunk_scores), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # Top 3 chunks per document
            
            processed_doc = {
                "original": doc,
                "chunks": [chunk for chunk, _ in top_chunks],
                "relevance_scores": [score for _, score in top_chunks],
                "combined_text": "\n".join([chunk for chunk, _ in top_chunks])
            }
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks with overlap"""
        tokens = self.synthesis_tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_length, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.synthesis_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - self.chunk_overlap
        
        return chunks
    
    async def _score_chunk_relevance(self, chunks: List[str], query: str) -> List[float]:
        """Score relevance of text chunks to query"""
        if not chunks:
            return []
        
        # Generate embeddings
        chunk_embeddings = self.sentence_model.encode(chunks)
        query_embedding = self.sentence_model.encode([query])
        
        # Calculate cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        
        return similarities.tolist()
    
    def _group_similar_documents(self, processed_docs: List[Dict]) -> List[List[Dict]]:
        """Group documents by semantic similarity"""
        if len(processed_docs) <= 2:
            return [processed_docs]
        
        # Generate embeddings for each document
        doc_texts = [doc["combined_text"] for doc in processed_docs]
        embeddings = self.sentence_model.encode(doc_texts)
        
        # Simple clustering based on similarity threshold
        groups = []
        used_indices = set()
        
        for i, embedding in enumerate(embeddings):
            if i in used_indices:
                continue
            
            # Find similar documents
            group = [processed_docs[i]]
            used_indices.add(i)
            
            for j, other_embedding in enumerate(embeddings):
                if j in used_indices or i == j:
                    continue
                
                similarity = np.dot(embedding, other_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                )
                
                if similarity > 0.7:  # Similarity threshold
                    group.append(processed_docs[j])
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    async def _generate_comprehensive_review(
        self, 
        doc_groups: List[List[Dict]], 
        query: str, 
        max_length: int
    ) -> str:
        """Generate comprehensive literature review"""
        sections = []
        
        # Introduction
        intro = await self._generate_introduction(query, doc_groups)
        sections.append(f"## Introduction\n\n{intro}")
        
        # Main synthesis by topic groups
        for i, group in enumerate(doc_groups):
            if len(doc_groups) > 1:
                topic = await self._identify_group_topic(group)
                sections.append(f"## {topic}\n")
            
            group_synthesis = await self._synthesize_document_group(group, query)
            sections.append(group_synthesis)
        
        # Conclusion and future directions
        conclusion = await self._generate_conclusion(doc_groups, query)
        sections.append(f"## Conclusion\n\n{conclusion}")
        
        return "\n\n".join(sections)
    
    async def _generate_comparison(
        self, 
        doc_groups: List[List[Dict]], 
        query: str, 
        max_length: int
    ) -> str:
        """Generate comparative analysis"""
        if len(doc_groups) < 2:
            return await self._generate_summary(doc_groups, query, max_length)
        
        sections = []
        
        # Extract key aspects for comparison
        aspects = await self._extract_comparison_aspects(doc_groups, query)
        
        sections.append(f"## Comparative Analysis: {query}\n")
        
        for aspect in aspects:
            section_text = f"### {aspect}\n\n"
            
            # Compare across groups
            comparisons = []
            for group in doc_groups:
                group_perspective = await self._extract_group_perspective(group, aspect)
                if group_perspective:
                    comparisons.append(group_perspective)
            
            if comparisons:
                comparison_text = await self._synthesize_comparisons(comparisons, aspect)
                section_text += comparison_text
                sections.append(section_text)
        
        return "\n\n".join(sections)
    
    async def _generate_summary(
        self, 
        doc_groups: List[List[Dict]], 
        query: str, 
        max_length: int
    ) -> str:
        """Generate concise summary"""
        all_docs = [doc for group in doc_groups for doc in group]
        
        # Extract key information from all documents
        key_points = []
        for doc in all_docs:
            points = await self._extract_key_points(doc, query)
            key_points.extend(points)
        
        # Remove duplicates and rank by importance
        unique_points = list(set(key_points))
        ranked_points = await self._rank_key_points(unique_points, query)
        
        # Generate coherent summary
        summary_prompt = f"""
        Based on the following key points about '{query}', write a coherent summary:
        
        {chr(10).join(f"- {point}" for point in ranked_points[:10])}
        
        Summary:
        """
        
        if self.config.openai_api_key:
            summary = await self._generate_with_openai(
                summary_prompt, 
                max_tokens=max_length
            )
        else:
            summary = await self._generate_with_bart(
                summary_prompt, 
                max_length=max_length
            )
        
        return summary
    
    async def _generate_with_openai(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate text using OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert academic researcher and writer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for more factual content
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            # Fallback to local model
            return await self._generate_with_bart(prompt, max_tokens)
    
    async def _generate_with_bart(self, prompt: str, max_length: int = 1500) -> str:
        """Generate text using local BART model"""
        inputs = self.synthesis_tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=self.max_input_length, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.synthesis_model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        generated_text = self.synthesis_tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    async def _insert_citations(self, text: str, citations: List[Citation]) -> str:
        """Insert citations into synthesized text"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated citation placement
        
        sentences = text.split('. ')
        cited_sentences = []
        
        for i, sentence in enumerate(sentences):
            cited_sentence = sentence
            
            # Add citations to sentences that reference specific findings
            if any(keyword in sentence.lower() for keyword in [
                'study', 'research', 'found', 'showed', 'demonstrated', 
                'according to', 'reported', 'observed'
            ]):
                # Find most relevant citation
                sentence_embedding = self.sentence_model.encode([sentence])
                
                best_citation = None
                best_similarity = 0
                
                for citation in citations:
                    citation_text = f"{citation.title} {citation.abstract}"
                    citation_embedding = self.sentence_model.encode([citation_text])
                    
                    similarity = np.dot(sentence_embedding, citation_embedding.T)[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_citation = citation
                
                if best_citation and best_similarity > 0.3:
                    cited_sentence += f" ({best_citation.formatted_citation})"
            
            cited_sentences.append(cited_sentence)
        
        return '. '.join(cited_sentences)
    
    async def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from synthesized text"""
        # Use NLP to identify key findings
        findings = []
        
        sentences = text.split('. ')
        for sentence in sentences:
            # Look for sentences that contain findings
            if any(indicator in sentence.lower() for indicator in [
                'found that', 'showed that', 'demonstrated that', 'revealed that',
                'indicated that', 'suggested that', 'concluded that'
            ]):
                findings.append(sentence.strip())
        
        return findings[:5]  # Return top 5 findings
    
    async def _identify_knowledge_gaps(
        self, 
        processed_docs: List[Dict], 
        query: str
    ) -> List[str]:
        """Identify gaps in the current knowledge"""
        gaps = []
        
        # Extract research questions and limitations from papers
        all_text = " ".join([doc["combined_text"] for doc in processed_docs])
        
        # Look for limitation statements
        limitation_indicators = [
            'limitation', 'further research', 'future work', 'not addressed',
            'remains unclear', 'unknown', 'unexplored', 'need for'
        ]
        
        sentences = all_text.split('. ')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in limitation_indicators):
                gaps.append(sentence.strip())
        
        return gaps[:3]  # Return top 3 gaps
    
    async def _find_contradictions(self, processed_docs: List[Dict]) -> List[str]:
        """Find contradictions between documents"""
        contradictions = []
        
        # Compare claims across documents
        doc_claims = []
        for doc in processed_docs:
            claims = await self._extract_claims(doc["combined_text"])
            doc_claims.append((doc["original"].title, claims))
        
        # Look for contradictory claims
        for i, (title1, claims1) in enumerate(doc_claims):
            for j, (title2, claims2) in enumerate(doc_claims[i+1:], i+1):
                for claim1 in claims1:
                    for claim2 in claims2:
                        if await self._are_contradictory(claim1, claim2):
                            contradiction = f"Contradiction between '{title1}' and '{title2}': {claim1} vs {claim2}"
                            contradictions.append(contradiction)
        
        return contradictions[:3]  # Return top 3 contradictions
    
    async def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simplified claim extraction
        claims = []
        sentences = text.split('. ')
        
        for sentence in sentences:
            # Look for sentences with strong factual indicators
            if any(indicator in sentence.lower() for indicator in [
                'is', 'are', 'shows', 'demonstrates', 'proves', 'indicates'
            ]) and len(sentence.split()) > 5:
                claims.append(sentence.strip())
        
        return claims[:5]  # Limit claims per document
    
    async def _are_contradictory(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are contradictory"""
        # Use semantic similarity and negation detection
        embeddings = self.sentence_model.encode([claim1, claim2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # High semantic similarity but potential negation indicates contradiction
        if similarity > 0.7:
            # Check for negation patterns
            negation_patterns = ['not', 'no', 'never', 'cannot', 'does not', 'did not']
            claim1_has_negation = any(pattern in claim1.lower() for pattern in negation_patterns)
            claim2_has_negation = any(pattern in claim2.lower() for pattern in negation_patterns)
            
            # Contradiction if one has negation and the other doesn't
            return claim1_has_negation != claim2_has_negation
        
        return False
    
    def _calculate_confidence_score(
        self, 
        processed_docs: List[Dict], 
        synthesis_text: str
    ) -> float:
        """Calculate confidence score for the synthesis"""
        factors = []
        
        # Number of sources
        source_score = min(1.0, len(processed_docs) / 10)
        factors.append(source_score)
        
        # Average citation count of sources
        citation_counts = [doc["original"].citation_count for doc in processed_docs]
        if citation_counts:
            avg_citations = np.mean(citation_counts)
            citation_score = min(1.0, avg_citations / 100)
            factors.append(citation_score)
        
        # Recency of sources
        current_year = datetime.now().year
        years = [doc["original"].publication_date.year for doc in processed_docs]
        if years:
            avg_year = np.mean(years)
            recency_score = max(0, min(1.0, (avg_year - 2015) / (current_year - 2015)))
            factors.append(recency_score)
        
        # Length and coherence of synthesis
        word_count = len(synthesis_text.split())
        length_score = min(1.0, word_count / 1000)  # Optimal around 1000 words
        factors.append(length_score)
        
        return np.mean(factors) if factors else 0.5

# Example usage
async def main():
    """Example usage of the LLM Synthesizer"""
    from .search_engine import AcademicSearchEngine
    from ..utils.config import Config
    
    config = Config()
    synthesizer = LLMSynthesizer(config)
    
    # Mock search results for demonstration
    mock_results = [
        SearchResult(
            title="Advances in Neuromorphic Computing",
            abstract="This paper reviews recent advances in neuromorphic computing...",
            authors=["Smith, J.", "Doe, A."],
            publication_date=datetime(2023, 1, 1),
            source="semantic_scholar",
            url="https://example.com/paper1",
            citation_count=45
        ),
        SearchResult(
            title="Brain-inspired Computing Architectures",
            abstract="We present novel brain-inspired architectures for computing...",
            authors=["Johnson, K.", "Brown, L."],
            publication_date=datetime(2022, 6, 15),
            source="arxiv",
            url="https://example.com/paper2",
            citation_count=23
        )
    ]
    
    # Synthesize information
    result = await synthesizer.synthesize(
        documents=mock_results,
        query="neuromorphic computing applications",
        synthesis_type="comprehensive_review",
        citation_style="apa"
    )
    
    print("Synthesis Result:")
    print(f"Word count: {result.word_count}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"Key findings: {len(result.key_findings)}")
    print(f"Sources used: {len(result.sources_used)}")
    print("\nSynthesized text:")
    print(result.synthesized_text[:500] + "...")

if __name__ == "__main__":
    asyncio.run(main())