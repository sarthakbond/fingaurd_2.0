"""
FinGuard 2.0 — Real-Time Investor Fraud & Impersonation Detection System
Phase 1: Working Application Foundation
"""
import sys
import uvicorn
from src.core.config import config
from src.core.logging import logger

def main():
    host = config.server.get("host", "127.0.0.1")
    port = int(config.server.get("port", 8000))
    
    logger.info("=================================================================")
    logger.info("   Starting FinGuard 2.0 Application Foundation Server")
    logger.info("   Problem Statement #7: AI-Based Investor Fraud Detection")
    logger.info(f"   API Docs: http://{host}:{port}/docs")
    logger.info(f"   Web Dashboard: http://{host}:{port}/")
    logger.info("=================================================================")
    
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
