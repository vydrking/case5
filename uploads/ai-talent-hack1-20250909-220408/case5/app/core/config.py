from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AutoReview Service"
    yandex_api_key: str | None = None
    yandex_folder_id: str | None = None
    yandex_model: str = "yandexgpt-lite"

    class Config:
        env_file = ".env"


settings = Settings()


