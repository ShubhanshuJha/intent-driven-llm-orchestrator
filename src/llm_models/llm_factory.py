from langchain_ollama import ChatOllama
from logger import get_logger, Logger


logger: Logger = None


LLM_PROFILE_CONFIGS: dict = {
    "agent": {
        "backend": "ollama",
        "model": "qwen3:8b",
        "temperature": 0.1,
        "max_tokens": 1024,
        "num_ctx": 8192,
        "reasoning": False,
        "keep_alive": -1
    },
    "general_chat": {
        "backend": "ollama",
        "model": "qwen3:30b",
        "temperature": 0.6,
        "max_tokens": 2048,
        "num_ctx": 8192,
        "reasoning": True,
        "keep_alive": -1
    },
    "coding": {
        "backend": "ollama",
        "model": "qwen3-coder:30b",
        "temperature": 0.2,
        "max_tokens": 8192,
        "num_ctx": 32768,
        "reasoning": False,
        "keep_alive": -1
    },
    "fast_lightweight": {
        "backend": "ollama",
        "model": "mistral:7b",
        "temperature": 0.5,
        "max_tokens": 512,
        "num_ctx": 4096,
        "reasoning": False,
        "keep_alive": -1
    },
    "judge": {
        "backend": "ollama",
        "model": "gpt-oss:20b",
        "temperature": 0.0,
        "max_tokens": 2048,
        "num_ctx": 32768,
        "reasoning": False,  # "high",
        "keep_alive": -1
    }
}


class LLMFactory:

    @staticmethod
    def get_model(profile: str, log_level: str = "INFO"):
        global logger

        if not logger:
            logger = get_logger(__file__, level=log_level)

        logger.info(
            f"LLM model requested | profile={profile}"
        )

        if profile not in LLM_PROFILE_CONFIGS:
            logger.error(
                f"Unsupported LLM profile requested | "
                f"profile={profile} | "
                f"available_profiles={list(LLM_PROFILE_CONFIGS.keys())}"
            )
            raise ValueError(
                f"(*) Unsupported Profile - {profile}!"
            )

        model_config = LLM_PROFILE_CONFIGS[profile]

        logger.debug(
            f"LLM profile resolved | "
            f"profile={profile} | "
            f"backend={model_config.get('backend')} | "
            f"model={model_config.get('model')}"
        )

        try:
            model = LLMFactory.__build_model(
                config=model_config,
                profile=profile
            )

            logger.info(
                f"LLM model initialized successfully | "
                f"profile={profile} | "
                f"backend={model_config.get('backend')} | "
                f"model={model_config.get('model')}"
            )

            return model

        except Exception as exc:
            logger.error(
                f"Failed to initialize LLM model | "
                f"profile={profile} | "
                f"error={exc}"
            )
            raise

    @staticmethod
    def __build_model(config: dict, profile: str):
        backend = config.get("backend")

        logger.debug(
            f"Building LLM model | "
            f"profile={profile} | "
            f"backend={backend} | "
            f"model={config.get('model')}"
        )

        if backend != "ollama":
            logger.error(
                f"Unsupported LLM backend | "
                f"profile={profile} | "
                f"backend={backend}"
            )

            raise ValueError(
                f"(*) Unsupported LLM Backend - {backend}!"
            )

        logger.debug(
            f"Initializing ChatOllama | "
            f"model={config['model']} | "
            f"temperature={config.get('temperature', 0.0)} | "
            f"max_tokens={config.get('max_tokens', 1024)} | "
            f"num_ctx={config.get('num_ctx', 8192)} | "
            f"reasoning={config.get('reasoning', False)} | "
            f"keep_alive={config.get('keep_alive', -1)}"
        )

        model = ChatOllama(
            model=config["model"],
            temperature=config.get("temperature", 0.0),
            num_predict=config.get("max_tokens", 1024),
            num_ctx=config.get("num_ctx", 8192),
            reasoning=config.get("reasoning", False),
            keep_alive=config.get("keep_alive", -1),
        )

        return model
