from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.review import router as review_router
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title="AutoReview Service", version="0.1.0")

    application.include_router(health_router, prefix="/api")
    application.include_router(review_router, prefix="/api")

    return application


app = create_app()


