#!/usr/bin/env python3
"""
Telegram Bot Integration
Отправка текста из файла в приватный чат через Telegram Bot API
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import aiohttp
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramError(Exception):
    """Базовое исключение для ошибок Telegram API."""
    pass


class TelegramAPI:
    """
    Асинхронный клиент для Telegram Bot API.
    Расширяемая архитектура с поддержкой разных методов.
    """

    API_BASE = "https://api.telegram.org/bot{token}/{method}"
    MAX_MESSAGE_LENGTH = 4096  # Лимит Telegram

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, **kwargs) -> dict:
        """Базовый метод для API-запросов с обработкой ошибок."""
        url = self.API_BASE.format(token=self.bot_token, method=method)

        try:
            async with self.session.post(url, data=kwargs) as response:
                result = await response.json()

                if not result.get('ok'):
                    error_code = result.get('error_code', 'Unknown')
                    description = result.get('description', 'No description')
                    raise TelegramError(f"API Error {error_code}: {description}")

                return result['result']

        except aiohttp.ClientError as e:
            raise TelegramError(f"Network error: {e}")

    async def send_message(
            self,
            chat_id: int | str,
            text: str,
            parse_mode: Optional[str] = None,
            disable_web_page_preview: bool = True
    ) -> dict:
        """
        Отправка текстового сообщения.
        Автоматически разбивает длинные сообщения.
        """
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            return await self._request(
                'sendMessage',
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )

        # Разбиваем длинные сообщения на части
        parts = self._split_message(text)
        results = []
        for part in parts:
            result = await self.send_message(
                chat_id=chat_id,
                text=part,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )
            results.append(result)
            await asyncio.sleep(0.5)  # Rate limiting
        return results[-1] if results else {}

    def _split_message(self, text: str, max_length: int = None) -> list[str]:
        """Разбиение текста на части с сохранением целостности параграфов."""
        max_length = max_length or self.MAX_MESSAGE_LENGTH
        if len(text) <= max_length:
            return [text]

        parts = []
        current_part = ""

        for paragraph in text.split('\n\n'):
            if len(current_part) + len(paragraph) + 2 <= max_length:
                current_part += paragraph + '\n\n'
            else:
                if current_part:
                    parts.append(current_part.strip())
                current_part = paragraph + '\n\n'

        if current_part:
            parts.append(current_part.strip())

        return parts

    async def get_me(self) -> dict:
        """Проверка валидности токена и получение информации о боте."""
        return await self._request('getMe')


class FileMessageSender:
    """
    High-level интерфейс для отправки содержимого файла.
    Легко расширяется для поддержки других форматов (Markdown, HTML).
    """

    def __init__(self, bot_token: str, chat_id: int | str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send_file_content(
            self,
            file_path: Path,
            parse_mode: Optional[str] = None,
            encoding: str = 'utf-8'
    ) -> bool:
        """Чтение файла и отправка его содержимого."""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")

            text = file_path.read_text(encoding=encoding)

            if not text.strip():
                logger.warning("Файл пустой")
                return False

            async with TelegramAPI(self.bot_token) as api:
                # Проверка подключения
                me = await api.get_me()
                logger.info(f"Бот @{me['username']} подключен")

                # Отправка
                result = await api.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=parse_mode
                )

                message_id = result.get('message_id')
                logger.info(f"Сообщение отправлено (ID: {message_id})")
                return True

        except TelegramError as e:
            logger.error(f"Ошибка Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return False


def validate_chat_id(chat_id: str) -> int | str:
    """
    Валидация chat_id.
    Поддерживает числовые ID и @username.
    """
    if chat_id.startswith('@'):
        return chat_id

    try:
        return int(chat_id)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Некорректный chat_id: {chat_id}. "
            "Используйте числовой ID или @username"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Отправка содержимого файла в Telegram"
    )
    parser.add_argument('file', type=Path, help='Путь к .txt файлу')
    parser.add_argument(
        '--chat-id', '-c',
        type=validate_chat_id,
        required=True,
        help='ID чата (число) или @username'
    )
    parser.add_argument(
        '--token', '-t',
        default=os.getenv('TELEGRAM_BOT_TOKEN'),
        help='Токен бота (или env TELEGRAM_BOT_TOKEN)'
    )
    parser.add_argument(
        '--parse-mode', '-p',
        choices=['Markdown', 'HTML', 'MarkdownV2'],
        help='Режим форматирования текста'
    )
    parser.add_argument(
        '--encoding', '-e',
        default='utf-8',
        help='Кодировка файла (default: utf-8)'
    )

    args = parser.parse_args()

    if not args.token:
        print("Ошибка: Укажите токен через --token или TELEGRAM_BOT_TOKEN", file=sys.stderr)
        sys.exit(1)

    sender = FileMessageSender(args.token, args.chat_id)
    success = asyncio.run(sender.send_file_content(
        file_path=args.file,
        parse_mode=args.parse_mode,
        encoding=args.encoding
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()