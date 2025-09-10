from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "AutoReview Service"
    yandex_api_key: Optional[str] = None
    yandex_folder_id: Optional[str] = None
    yandex_model: str = "yandexgpt-lite"

    class Config:
        env_file = ".env"

    def validate_yandex_credentials(self) -> None:
        """Validate that Yandex credentials are present when needed."""
        if self.yandex_api_key is None and self.yandex_folder_id is None:
            # Offline mode - no validation needed
            return

        if self.yandex_api_key is None:
            raise ValueError("YANDEX_API_KEY is required when using YandexGPT features")

        if self.yandex_folder_id is None:
            raise ValueError("YANDEX_FOLDER_ID is required when using YandexGPT features")

        # Validate credential format
        if not self.yandex_api_key.strip():
            raise ValueError("YANDEX_API_KEY cannot be empty")

        if not self.yandex_folder_id.strip():
            raise ValueError("YANDEX_FOLDER_ID cannot be empty")


settings = Settings()


