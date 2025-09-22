"""
Follow-up Question Generation System
Generates contextual follow-up questions for iterative research
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import openai
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, T5TokenizerFast, T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import spacy

from .llm_synthesizer import SynthesisResult
from .search_engine import SearchResult
from ..utils.config import Config

@dataclass
class FollowUpQuestion:
    """Represents a generated follow-up question"""
    question: str
    question_type: str
    confidence_score: float
    context_relevance: float
    complexity_level: str
    knowledge_gap: str
    suggested_sources: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "question_type": self.question_type,
            "confidence_score": self.confidence_score,
            "context_relevance": self.context_relevance,
            "complexity_level": self.complexity_level,
            "knowledge_gap": self.knowledge_gap,
            "suggested_sources": self.suggested_sources
        }

@dataclass
class QuestionGenerationResult:
    """Result of follow-up question generation"""
    questions: List[FollowUpQuestion]
    conversation_context: str
    knowledge_gaps_identified: List[str]
    total_questions: int
    avg_confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "conversation_context": self.conversation_context,
            "knowledge_gaps_identified": self.knowledge_gaps_identified,
            "total_questions": self.total_questions,
            "avg_confidence": self.avg_confidence
        }

class FollowUpGenerator:
    """
    Advanced follow-up question generator that creates contextual,
    relevant questions for iterative research exploration
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._init_models()
        
        # Question generation parameters
        self.max_questions = 5
        self.min_confidence_threshold = 0.6
        self.context_window_size = 2000  # tokens
        
        # Question types and their strategies
        self.question_types = {
            "clarification": "Questions that seek to clarify unclear or ambiguous points",
            "expansion": "Questions that explore related topics or deeper aspects",
            "comparison": "Questions that compare different approaches or findings",
            "application": "Questions about practical applications or implementations",
            "methodology": "Questions about research methods or experimental design",
            "limitation": "Questions about limitations or constraints",
            "future_work": "Questions about future research directions",
            "causation": "Questions about cause-and-effect relationships",
            "evaluation": "Questions about evaluation metrics or criteria"
        }
        
    def _init_models(self):
        """Initialize models for question generation"""
        # Primary question generation model (T5 fine-tuned for question generation)
        self.question_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
        self.question_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Sentence transformer for context understanding
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        # NLP pipeline for text analysis
        self.nlp = spacy.load('en_core_web_sm')
        
        # OpenAI client for advanced question generation
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        
        # Context analyzer for conversation understanding
        self.context_analyzer = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            return_all_scores=True
        )
    
    async def generate_follow_up_questions(
        self,
        synthesis_result: SynthesisResult,
        original_query: str,
        conversation_history: List[str] = None,
        question_types: List[str] = None,
        max_questions: int = 3,
        complexity_level: str = "intermediate"
    ) -> QuestionGenerationResult:
        """
        Generate contextual follow-up questions based on synthesis results
        
        Args:
            synthesis_result: Result from previous synthesis
            original_query: The original research query
            conversation_history: Previous questions and answers
            question_types: Types of questions to generate
            max_questions: Maximum number of questions to generate
            complexity_level: Complexity level ('basic', 'intermediate', 'advanced')
            
        Returns:
            QuestionGenerationResult with generated questions and metadata
        """
        self.logger.info(f"Generating follow-up questions for: {original_query}")
        
        if question_types is None:
            question_types = ["clarification", "expansion", "application"]
        
        # Build conversation context
        context = self._build_conversation_context(
            synthesis_result, original_query, conversation_history
        )
        
        # Identify knowledge gaps
        knowledge_gaps = await self._identify_knowledge_gaps(
            synthesis_result, original_query
        )
        
        # Generate questions for each requested type
        all_questions = []
        for question_type in question_types:
            questions = await self._generate_questions_by_type(
                context, question_type, knowledge_gaps, complexity_level
            )
            all_questions.extend(questions)
        
        # Rank and filter questions
        ranked_questions = await self._rank_and_filter_questions(
            all_questions, context, max_questions
        )
        
        # Calculate metrics
        avg_confidence = np.mean([q.confidence_score for q in ranked_questions]) if ranked_questions else 0
        
        return QuestionGenerationResult(
            questions=ranked_questions,
            conversation_context=context,
            knowledge_gaps_identified=knowledge_gaps,
            total_questions=len(ranked_questions),
            avg_confidence=avg_confidence
        )
    
    def _build_conversation_context(
        self,
        synthesis_result: SynthesisResult,
        original_query: str,
        conversation_history: List[str] = None
    ) -> str:
        """Build comprehensive conversation context"""
        context_parts = []
        
        # Add original query
        context_parts.append(f"Original Query: {original_query}")
        
        # Add synthesis summary
        context_parts.append(f"Synthesis Summary: {synthesis_result.synthesized_text[:500]}...")
        
        # Add key findings
        if synthesis_result.key_findings:
            context_parts.append("Key Findings:")
            for finding in synthesis_result.key_findings:
                context_parts.append(f"- {finding}")
        
        # Add identified gaps
        if synthesis_result.gaps_identified:
            context_parts.append("Knowledge Gaps:")
            for gap in synthesis_result.gaps_identified:
                context_parts.append(f"- {gap}")
        
        # Add contradictions if any
        if synthesis_result.contradictions:
            context_parts.append("Contradictions Found:")
            for contradiction in synthesis_result.contradictions:
                context_parts.append(f"- {contradiction}")
        
        # Add conversation history
        if conversation_history:
            context_parts.append("Previous Questions:")
            for i, item in enumerate(conversation_history[-5:]):  # Last 5 items
                context_parts.append(f"{i+1}. {item}")
        
        return "\n\n".join(context_parts)
    
    async def _identify_knowledge_gaps(
        self,
        synthesis_result: SynthesisResult,
        original_query: str
    ) -> List[str]:
        """Identify knowledge gaps from synthesis result"""
        gaps = []
        
        # Use existing gaps from synthesis
        gaps.extend(synthesis_result.gaps_identified)
        
        # Analyze text for implicit gaps
        doc = self.nlp(synthesis_result.synthesized_text)
        
        # Look for uncertainty markers
        uncertainty_patterns = [
            "unclear", "unknown", "not well understood", "remains to be seen",
            "further research needed", "more studies required", "limited evidence"
        ]
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for pattern in uncertainty_patterns:
                if pattern in sent_text:
                    gaps.append(sent.text.strip())
                    break
        
        # Look for questions in the text (unanswered questions)
        for sent in doc.sents:
            if sent.text.strip().endswith('?'):
                gaps.append(f"Unanswered question: {sent.text.strip()}")
        
        return list(set(gaps))  # Remove duplicates
    
    async def _generate_questions_by_type(
        self,
        context: str,
        question_type: str,
        knowledge_gaps: List[str],
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions of a specific type"""
        questions = []
        
        if question_type == "clarification":
            questions = await self._generate_clarification_questions(
                context, knowledge_gaps, complexity_level
            )
        elif question_type == "expansion":
            questions = await self._generate_expansion_questions(
                context, complexity_level
            )
        elif question_type == "comparison":
            questions = await self._generate_comparison_questions(
                context, complexity_level
            )
        elif question_type == "application":
            questions = await self._generate_application_questions(
                context, complexity_level
            )
        elif question_type == "methodology":
            questions = await self._generate_methodology_questions(
                context, complexity_level
            )
        elif question_type == "limitation":
            questions = await self._generate_limitation_questions(
                context, knowledge_gaps, complexity_level
            )
        elif question_type == "future_work":
            questions = await self._generate_future_work_questions(
                context, knowledge_gaps, complexity_level
            )
        elif question_type == "causation":
            questions = await self._generate_causation_questions(
                context, complexity_level
            )
        elif question_type == "evaluation":
            questions = await self._generate_evaluation_questions(
                context, complexity_level
            )
        
        return questions
    
    async def _generate_clarification_questions(
        self,
        context: str,
        knowledge_gaps: List[str],
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate clarification questions"""
        questions = []
        
        # Extract unclear terms and concepts
        doc = self.nlp(context)
        technical_terms = []
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                len(token.text) > 3 and 
                token.text[0].isupper()):
                technical_terms.append(token.text)
        
        # Generate questions about technical terms
        for term in set(technical_terms[:5]):  # Top 5 unique terms
            question_text = await self._generate_term_clarification_question(
                term, context, complexity_level
            )
            if question_text:
                question = FollowUpQuestion(
                    question=question_text,
                    question_type="clarification",
                    confidence_score=0.8,
                    context_relevance=0.9,
                    complexity_level=complexity_level,
                    knowledge_gap=f"Understanding of term: {term}",
                    suggested_sources=["technical dictionaries", "academic papers"]
                )
                questions.append(question)
        
        # Generate questions about knowledge gaps
        for gap in knowledge_gaps[:3]:  # Top 3 gaps
            question_text = await self._generate_gap_clarification_question(
                gap, complexity_level
            )
            if question_text:
                question = FollowUpQuestion(
                    question=question_text,
                    question_type="clarification",
                    confidence_score=0.7,
                    context_relevance=0.8,
                    complexity_level=complexity_level,
                    knowledge_gap=gap,
                    suggested_sources=["recent research papers", "review articles"]
                )
                questions.append(question)
        
        return questions
    
    async def _generate_expansion_questions(
        self,
        context: str,
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions that explore related topics"""
        questions = []
        
        # Extract key concepts for expansion
        doc = self.nlp(context)
        key_concepts = []
        
        # Extract noun phrases as potential concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep it manageable
                key_concepts.append(chunk.text)
        
        # Generate expansion questions
        expansion_templates = [
            "What are the broader implications of {concept}?",
            "How does {concept} relate to other areas in the field?",
            "What are the different types or categories of {concept}?",
            "What are the historical developments in {concept}?",
            "How might {concept} evolve in the future?"
        ]
        
        for concept in set(key_concepts[:5]):
            template = np.random.choice(expansion_templates)
            question_text = template.format(concept=concept)
            
            question = FollowUpQuestion(
                question=question_text,
                question_type="expansion",
                confidence_score=0.75,
                context_relevance=0.7,
                complexity_level=complexity_level,
                knowledge_gap=f"Broader understanding of {concept}",
                suggested_sources=["comprehensive reviews", "survey papers"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_comparison_questions(
        self,
        context: str,
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate comparative questions"""
        questions = []
        
        # Look for comparisons already mentioned in context
        doc = self.nlp(context)
        comparison_indicators = ["versus", "compared to", "different from", "similar to", "unlike"]
        
        comparison_contexts = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(indicator in sent_text for indicator in comparison_indicators):
                comparison_contexts.append(sent.text)
        
        # Generate comparison questions
        comparison_templates = [
            "How do different approaches to this problem compare in terms of effectiveness?",
            "What are the advantages and disadvantages of the main methods discussed?",
            "How do the findings from different studies compare or contrast?",
            "What are the trade-offs between different solutions?",
            "Which approach would be most suitable for different scenarios?"
        ]
        
        for template in comparison_templates[:3]:  # Generate 3 comparison questions
            question = FollowUpQuestion(
                question=template,
                question_type="comparison",
                confidence_score=0.7,
                context_relevance=0.75,
                complexity_level=complexity_level,
                knowledge_gap="Comparative analysis of approaches",
                suggested_sources=["comparative studies", "meta-analyses"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_application_questions(
        self,
        context: str,
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions about practical applications"""
        questions = []
        
        application_templates = [
            "What are the real-world applications of these findings?",
            "How could this research be implemented in practice?",
            "What are the practical challenges in applying this knowledge?",
            "What tools or resources would be needed for implementation?",
            "What are some successful case studies or examples?",
            "How can organizations benefit from these insights?",
            "What are the cost-benefit considerations?",
            "What skills or expertise would be required for implementation?"
        ]
        
        for template in application_templates[:4]:  # Generate 4 application questions
            question = FollowUpQuestion(
                question=template,
                question_type="application",
                confidence_score=0.8,
                context_relevance=0.8,
                complexity_level=complexity_level,
                knowledge_gap="Practical implementation guidance",
                suggested_sources=["case studies", "implementation guides", "industry reports"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_methodology_questions(
        self,
        context: str,
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions about research methodology"""
        questions = []
        
        methodology_templates = [
            "What research methodologies were used in these studies?",
            "How were the data collected and analyzed?",
            "What are the strengths and limitations of the research methods used?",
            "How could the methodology be improved or extended?",
            "What alternative research approaches could be used?",
            "How was validity and reliability ensured in these studies?",
            "What ethical considerations were addressed?",
            "How representative are the samples used?"
        ]
        
        for template in methodology_templates[:3]:  # Generate 3 methodology questions
            question = FollowUpQuestion(
                question=template,
                question_type="methodology",
                confidence_score=0.75,
                context_relevance=0.7,
                complexity_level=complexity_level,
                knowledge_gap="Understanding of research methods",
                suggested_sources=["methodology papers", "research design guides"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_limitation_questions(
        self,
        context: str,
        knowledge_gaps: List[str],
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions about limitations"""
        questions = []
        
        limitation_templates = [
            "What are the main limitations of the current research?",
            "What factors might affect the generalizability of these findings?",
            "What assumptions were made that might not hold in all cases?",
            "What are the potential sources of bias in these studies?",
            "What contextual factors might influence the results?",
            "What are the boundaries of applicability for these findings?"
        ]
        
        for template in limitation_templates[:3]:  # Generate 3 limitation questions
            question = FollowUpQuestion(
                question=template,
                question_type="limitation",
                confidence_score=0.7,
                context_relevance=0.8,
                complexity_level=complexity_level,
                knowledge_gap="Understanding of research limitations",
                suggested_sources=["critical reviews", "discussion sections"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_future_work_questions(
        self,
        context: str,
        knowledge_gaps: List[str],
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions about future research directions"""
        questions = []
        
        future_work_templates = [
            "What are the most promising directions for future research?",
            "What questions remain unanswered in this field?",
            "How might emerging technologies change this area of research?",
            "What interdisciplinary approaches could advance this field?",
            "What would be the next logical steps in this research program?",
            "What new methodologies might be needed to address current limitations?"
        ]
        
        for template in future_work_templates[:3]:  # Generate 3 future work questions
            question = FollowUpQuestion(
                question=template,
                question_type="future_work",
                confidence_score=0.8,
                context_relevance=0.7,
                complexity_level=complexity_level,
                knowledge_gap="Future research directions",
                suggested_sources=["research proposals", "funding calls", "conference proceedings"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_causation_questions(
        self,
        context: str,
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions about cause-and-effect relationships"""
        questions = []
        
        causation_templates = [
            "What are the underlying causes of the phenomena described?",
            "How do different factors influence the outcomes?",
            "What are the direct versus indirect effects?",
            "What mechanisms explain the observed relationships?",
            "How do various variables interact with each other?",
            "What are the necessary versus sufficient conditions?"
        ]
        
        for template in causation_templates[:3]:  # Generate 3 causation questions
            question = FollowUpQuestion(
                question=template,
                question_type="causation",
                confidence_score=0.75,
                context_relevance=0.8,
                complexity_level=complexity_level,
                knowledge_gap="Causal understanding",
                suggested_sources=["causal inference studies", "mechanistic papers"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_evaluation_questions(
        self,
        context: str,
        complexity_level: str
    ) -> List[FollowUpQuestion]:
        """Generate questions about evaluation and assessment"""
        questions = []
        
        evaluation_templates = [
            "How should the success of these approaches be measured?",
            "What metrics or criteria are most appropriate for evaluation?",
            "How do different evaluation methods compare?",
            "What are the standards or benchmarks in this field?",
            "How can quality or effectiveness be assessed?",
            "What validation approaches are most suitable?"
        ]
        
        for template in evaluation_templates[:3]:  # Generate 3 evaluation questions
            question = FollowUpQuestion(
                question=template,
                question_type="evaluation",
                confidence_score=0.7,
                context_relevance=0.75,
                complexity_level=complexity_level,
                knowledge_gap="Evaluation methodology",
                suggested_sources=["evaluation frameworks", "benchmarking studies"]
            )
            questions.append(question)
        
        return questions
    
    async def _generate_term_clarification_question(
        self,
        term: str,
        context: str,
        complexity_level: str
    ) -> Optional[str]:
        """Generate a clarification question for a specific term"""
        templates = [
            f"What exactly is meant by '{term}' in this context?",
            f"Could you provide a more detailed explanation of '{term}'?",
            f"How is '{term}' defined or characterized in this field?",
            f"What are the key components or aspects of '{term}'?"
        ]
        
        return np.random.choice(templates)
    
    async def _generate_gap_clarification_question(
        self,
        gap: str,
        complexity_level: str
    ) -> Optional[str]:
        """Generate a clarification question for a knowledge gap"""
        if "unclear" in gap.lower():
            return f"Could you elaborate on what specifically is unclear: {gap}?"
        elif "unknown" in gap.lower():
            return f"What would need to be studied to address: {gap}?"
        elif "further research" in gap.lower():
            return f"What type of research would be most valuable for: {gap}?"
        else:
            return f"Could you provide more context about: {gap}?"
    
    async def _rank_and_filter_questions(
        self,
        questions: List[FollowUpQuestion],
        context: str,
        max_questions: int
    ) -> List[FollowUpQuestion]:
        """Rank and filter questions based on quality and relevance"""
        if not questions:
            return []
        
        # Calculate relevance scores
        context_embedding = self.sentence_model.encode([context])
        
        for question in questions:
            question_embedding = self.sentence_model.encode([question.question])
            relevance = np.dot(context_embedding, question_embedding.T)[0][0]
            question.context_relevance = float(relevance)
        
        # Calculate diversity scores (avoid too similar questions)
        question_embeddings = self.sentence_model.encode([q.question for q in questions])
        
        selected_questions = []
        selected_embeddings = []
        
        # Sort by combined score (confidence + relevance)
        questions.sort(key=lambda q: q.confidence_score + q.context_relevance, reverse=True)
        
        for question in questions:
            if len(selected_questions) >= max_questions:
                break
            
            # Check diversity with already selected questions
            question_embedding = self.sentence_model.encode([question.question])
            
            if not selected_embeddings:
                # First question
                selected_questions.append(question)
                selected_embeddings.append(question_embedding)
            else:
                # Check similarity with selected questions
                similarities = [
                    np.dot(question_embedding, selected_emb.T)[0][0]
                    for selected_emb in selected_embeddings
                ]
                max_similarity = max(similarities)
                
                # Only add if not too similar to existing questions
                if max_similarity < 0.8:  # Similarity threshold
                    selected_questions.append(question)
                    selected_embeddings.append(question_embedding)
        
        # Filter by minimum confidence threshold
        filtered_questions = [
            q for q in selected_questions 
            if q.confidence_score >= self.min_confidence_threshold
        ]
        
        return filtered_questions

# Example usage
async def main():
    """Example usage of the Follow-up Generator"""
    from .llm_synthesizer import SynthesisResult, SearchResult
    from ..utils.config import Config
    
    config = Config()
    generator = FollowUpGenerator(config)
    
    # Mock synthesis result
    mock_synthesis = SynthesisResult(
        synthesized_text="Recent advances in neuromorphic computing show promising applications in edge computing and AI acceleration. However, several challenges remain in terms of power efficiency and scalability.",
        sources_used=[],
        citations=[],
        confidence_score=0.8,
        synthesis_type="comprehensive_review",
        word_count=150,
        key_findings=["Power efficiency improvements", "Scalability challenges"],
        gaps_identified=["Long-term reliability unclear", "Cost-benefit analysis needed"],
        contradictions=[]
    )
    
    # Generate follow-up questions
    result = await generator.generate_follow_up_questions(
        synthesis_result=mock_synthesis,
        original_query="neuromorphic computing applications",
        question_types=["clarification", "expansion", "application"],
        max_questions=5,
        complexity_level="intermediate"
    )
    
    print(f"Generated {result.total_questions} follow-up questions:")
    for i, question in enumerate(result.questions, 1):
        print(f"\n{i}. {question.question}")
        print(f"   Type: {question.question_type}")
        print(f"   Confidence: {question.confidence_score:.3f}")
        print(f"   Relevance: {question.context_relevance:.3f}")
        print(f"   Gap: {question.knowledge_gap}")

if __name__ == "__main__":
    asyncio.run(main())