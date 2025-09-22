"""
FastAPI Application Routes
Main API endpoints for the Context-Aware Research Assistant
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from .schemas import (
    SearchRequest, SearchResponse, SynthesisRequest, SynthesisResponse,
    FollowUpRequest, FollowUpResponse, CitationRequest, CitationResponse,
    HealthResponse, UserSession, ErrorResponse
)
from .middleware import rate_limiter, authenticate_user, log_request
from ..core.search_engine import AcademicSearchEngine
from ..core.llm_synthesizer import LLMSynthesizer
from ..core.citation_manager import CitationManager
from ..core.follow_up_generator import FollowUpGenerator
from ..utils.config import Config
from ..utils.ethics_checker import EthicsChecker

# Initialize FastAPI app
app = FastAPI(
    title="Context-Aware AI Research Assistant",
    description="Advanced research assistant with academic search, synthesis, and citation generation",
    version="1.0.0",
    contact={
        "name": "Research Assistant Team",
        "email": "support@research-assistant.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Global instances (will be initialized on startup)
search_engine: Optional[AcademicSearchEngine] = None
synthesizer: Optional[LLMSynthesizer] = None
citation_manager: Optional[CitationManager] = None
follow_up_generator: Optional[FollowUpGenerator] = None
ethics_checker: Optional[EthicsChecker] = None

# Session storage (in production, use Redis or database)
active_sessions: Dict[str, UserSession] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global search_engine, synthesizer, citation_manager, follow_up_generator, ethics_checker
    
    logger.info("Starting up Context-Aware Research Assistant API")
    
    # Initialize core components
    search_engine = AcademicSearchEngine(config)
    synthesizer = LLMSynthesizer(config)
    citation_manager = CitationManager(config)
    follow_up_generator = FollowUpGenerator(config)
    ethics_checker = EthicsChecker()
    
    logger.info("All services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Context-Aware Research Assistant API")
    
    # Close any open connections
    if hasattr(search_engine, 'close'):
        await search_engine.close()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services={
            "search_engine": search_engine is not None,
            "synthesizer": synthesizer is not None,
            "citation_manager": citation_manager is not None,
            "follow_up_generator": follow_up_generator is not None
        }
    )

# Authentication dependency
async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserSession:
    """Get current user session"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    session_id = credentials.credentials
    if session_id not in active_sessions:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    session = active_sessions[session_id]
    if session.expires_at < datetime.now():
        del active_sessions[session_id]
        raise HTTPException(status_code=401, detail="Session expired")
    
    return session

# Session management
@app.post("/auth/login")
async def create_session(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Create a new user session"""
    session_id = str(uuid.uuid4())
    session = UserSession(
        session_id=session_id,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=24),
        queries_count=0,
        rate_limit_remaining=100
    )
    
    active_sessions[session_id] = session
    
    # Cleanup expired sessions in background
    background_tasks.add_task(cleanup_expired_sessions)
    
    return {"session_id": session_id, "expires_in": "24 hours"}

async def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session in active_sessions.items()
        if session.expires_at < current_time
    ]
    
    for session_id in expired_sessions:
        del active_sessions[session_id]
    
    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Main search endpoint
@app.post("/search", response_model=SearchResponse)
@rate_limiter
async def search_academic_papers(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_current_session)
):
    """
    Search academic papers and sources
    
    This endpoint searches across multiple academic databases including
    Semantic Scholar and ArXiv to find relevant research papers.
    """
    try:
        # Log request
        background_tasks.add_task(
            log_request, 
            session.session_id, 
            "search", 
            request.dict()
        )
        
        # Check ethics compliance
        if not ethics_checker.is_query_ethical(request.query):
            raise HTTPException(
                status_code=400, 
                detail="Query violates ethical guidelines"
            )
        
        # Update session
        session.queries_count += 1
        session.last_activity = datetime.now()
        
        # Perform search
        async with search_engine as engine:
            results = await engine.search(
                query=request.query,
                sources=request.sources,
                max_results=request.max_results,
                time_range=(request.start_date, request.end_date) if request.start_date and request.end_date else None,
                bias_filter=request.apply_bias_filter,
                citation_threshold=request.min_citations,
                fields_of_study=request.fields_of_study
            )
        
        # Convert to response format
        search_results = [result.to_dict() for result in results]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time=datetime.now(),
            sources_searched=request.sources,
            filters_applied={
                "bias_filter": request.apply_bias_filter,
                "min_citations": request.min_citations,
                "fields_of_study": request.fields_of_study,
                "date_range": [request.start_date, request.end_date] if request.start_date else None
            }
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Synthesis endpoint
@app.post("/synthesize", response_model=SynthesisResponse)
@rate_limiter
async def synthesize_documents(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_current_session)
):
    """
    Synthesize information from multiple documents
    
    This endpoint takes search results and synthesizes them into a coherent
    response with proper citations and analysis.
    """
    try:
        # Log request
        background_tasks.add_task(
            log_request,
            session.session_id,
            "synthesize",
            {"query": request.query, "document_count": len(request.document_ids)}
        )
        
        # Update session
        session.queries_count += 1
        session.last_activity = datetime.now()
        
        # Convert document IDs to SearchResult objects (simplified)
        # In production, you'd fetch from database or cache
        documents = []  # This would be populated from stored search results
        
        # Perform synthesis
        synthesis_result = await synthesizer.synthesize(
            documents=documents,
            query=request.query,
            synthesis_type=request.synthesis_type,
            citation_style=request.citation_style,
            max_length=request.max_length,
            include_contradictions=request.include_contradictions
        )
        
        return SynthesisResponse(
            synthesized_text=synthesis_result.synthesized_text,
            sources_used=[source.to_dict() for source in synthesis_result.sources_used],
            citations=[citation.to_dict() for citation in synthesis_result.citations],
            confidence_score=synthesis_result.confidence_score,
            synthesis_type=synthesis_result.synthesis_type,
            word_count=synthesis_result.word_count,
            key_findings=synthesis_result.key_findings,
            knowledge_gaps=synthesis_result.gaps_identified,
            contradictions=synthesis_result.contradictions,
            processing_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Citation generation endpoint
@app.post("/citations", response_model=CitationResponse)
@rate_limiter
async def generate_citations(
    request: CitationRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_current_session)
):
    """
    Generate citations for documents
    
    This endpoint generates properly formatted citations in multiple styles
    with verification of links and metadata enrichment.
    """
    try:
        # Log request
        background_tasks.add_task(
            log_request,
            session.session_id,
            "citations",
            {"document_count": len(request.document_ids), "style": request.citation_style}
        )
        
        # Update session
        session.queries_count += 1
        session.last_activity = datetime.now()
        
        # Convert document IDs to SearchResult objects
        documents = []  # This would be populated from stored search results
        
        # Generate citations
        async with citation_manager as cm:
            citations = await cm.generate_citations(
                sources=documents,
                citation_style=request.citation_style,
                verify_links=request.verify_links,
                enrich_metadata=request.enrich_metadata
            )
            
            # Generate bibliography if requested
            bibliography = ""
            if request.include_bibliography:
                bibliography = await cm.generate_bibliography(
                    citations=citations,
                    style=request.citation_style,
                    sort_alphabetically=True
                )
            
            # Validate citations
            validation_metrics = await cm.validate_citations(citations)
        
        return CitationResponse(
            citations=[citation.to_dict() for citation in citations],
            bibliography=bibliography,
            citation_style=request.citation_style,
            total_citations=len(citations),
            validation_metrics=validation_metrics,
            processing_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Citation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Follow-up questions endpoint
@app.post("/follow-up", response_model=FollowUpResponse)
@rate_limiter
async def generate_follow_up_questions(
    request: FollowUpRequest,
    background_tasks: BackgroundTasks,
    session: UserSession = Depends(get_current_session)
):
    """
    Generate contextual follow-up questions
    
    This endpoint generates intelligent follow-up questions based on
    the synthesis results and conversation context.
    """
    try:
        # Log request
        background_tasks.add_task(
            log_request,
            session.session_id,
            "follow_up",
            {"query": request.original_query, "question_types": request.question_types}
        )
        
        # Update session
        session.queries_count += 1
        session.last_activity = datetime.now()
        
        # Create mock synthesis result (in production, fetch from storage)
        synthesis_result = None  # This would be fetched based on synthesis_id
        
        # Generate follow-up questions
        question_result = await follow_up_generator.generate_follow_up_questions(
            synthesis_result=synthesis_result,
            original_query=request.original_query,
            conversation_history=request.conversation_history,
            question_types=request.question_types,
            max_questions=request.max_questions,
            complexity_level=request.complexity_level
        )
        
        return FollowUpResponse(
            questions=[q.to_dict() for q in question_result.questions],
            total_questions=question_result.total_questions,
            avg_confidence=question_result.avg_confidence,
            knowledge_gaps=question_result.knowledge_gaps_identified,
            conversation_context=question_result.conversation_context,
            processing_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Follow-up generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Research pipeline endpoint (combines multiple steps)
@app.post("/research")
async def comprehensive_research(
    query: str = Query(..., description="Research query"),
    max_results: int = Query(20, description="Maximum search results"),
    synthesis_type: str = Query("comprehensive_review", description="Type of synthesis"),
    citation_style: str = Query("apa", description="Citation style"),
    generate_follow_ups: bool = Query(True, description="Generate follow-up questions"),
    session: UserSession = Depends(get_current_session)
):
    """
    Comprehensive research pipeline
    
    This endpoint combines search, synthesis, citation generation, and follow-up
    question generation into a single comprehensive research workflow.
    """
    try:
        # Step 1: Search for papers
        async with search_engine as engine:
            search_results = await engine.search(
                query=query,
                max_results=max_results,
                bias_filter=True
            )
        
        # Step 2: Synthesize results
        synthesis_result = await synthesizer.synthesize(
            documents=search_results,
            query=query,
            synthesis_type=synthesis_type,
            citation_style=citation_style
        )
        
        # Step 3: Generate citations
        async with citation_manager as cm:
            citations = await cm.generate_citations(
                sources=search_results,
                citation_style=citation_style,
                verify_links=True
            )
        
        # Step 4: Generate follow-up questions (if requested)
        follow_up_questions = []
        if generate_follow_ups:
            question_result = await follow_up_generator.generate_follow_up_questions(
                synthesis_result=synthesis_result,
                original_query=query,
                max_questions=5
            )
            follow_up_questions = [q.to_dict() for q in question_result.questions]
        
        # Update session
        session.queries_count += 1
        session.last_activity = datetime.now()
        
        return {
            "query": query,
            "search_results": [result.to_dict() for result in search_results[:10]],  # Top 10
            "synthesis": {
                "text": synthesis_result.synthesized_text,
                "confidence": synthesis_result.confidence_score,
                "key_findings": synthesis_result.key_findings,
                "knowledge_gaps": synthesis_result.gaps_identified
            },
            "citations": [citation.to_dict() for citation in citations[:5]],  # Top 5
            "follow_up_questions": follow_up_questions,
            "processing_time": datetime.now().isoformat(),
            "sources_count": len(search_results),
            "citations_count": len(citations)
        }
        
    except Exception as e:
        logger.error(f"Research pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/session")
async def get_session_analytics(
    session: UserSession = Depends(get_current_session)
):
    """Get analytics for current session"""
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "queries_count": session.queries_count,
        "last_activity": session.last_activity,
        "rate_limit_remaining": session.rate_limit_remaining,
        "session_duration": (datetime.now() - session.created_at).total_seconds() / 3600  # hours
    }

@app.get("/analytics/system")
async def get_system_analytics():
    """Get system-wide analytics (admin only)"""
    total_sessions = len(active_sessions)
    active_sessions_count = sum(
        1 for session in active_sessions.values()
        if (datetime.now() - session.last_activity).seconds < 3600  # Active in last hour
    )
    
    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions_count,
        "services_status": {
            "search_engine": search_engine is not None,
            "synthesizer": synthesizer is not None,
            "citation_manager": citation_manager is not None,
            "follow_up_generator": follow_up_generator is not None
        },
        "timestamp": datetime.now()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return ErrorResponse(
        error=exc.detail,
        status_code=exc.status_code,
        timestamp=datetime.now()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="Internal server error",
        status_code=500,
        timestamp=datetime.now()
    )

# Additional utility endpoints
@app.get("/sources")
async def get_available_sources():
    """Get list of available search sources"""
    return {
        "sources": [
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "description": "AI-powered academic search engine",
                "paper_count": "200M+",
                "coverage": "Computer Science, Biology, Medicine, Physics"
            },
            {
                "id": "arxiv",
                "name": "arXiv",
                "description": "Repository of preprint papers",
                "paper_count": "2M+",
                "coverage": "Physics, Mathematics, Computer Science, Biology"
            }
        ]
    }

@app.get("/citation-styles")
async def get_citation_styles():
    """Get available citation styles"""
    return {
        "styles": [
            {"id": "apa", "name": "APA", "description": "American Psychological Association"},
            {"id": "mla", "name": "MLA", "description": "Modern Language Association"},
            {"id": "chicago", "name": "Chicago", "description": "Chicago Manual of Style"},
            {"id": "ieee", "name": "IEEE", "description": "Institute of Electrical and Electronics Engineers"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)