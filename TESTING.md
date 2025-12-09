# Инструкция по тестированию WhisperLive Unified Server

## Быстрый старт

### 1. Подготовка окружения

```bash
# Скопируйте пример конфигурации
cp .env.example .env

# Отредактируйте .env и установите свой API_KEY
nano .env

# Создайте директории для кэша
mkdir -p models whisper-cache
```

### 2. Запуск сервера

```bash
# Сборка и запуск контейнера
docker-compose up -d

# Просмотр логов
docker-compose logs -f
```

### 3. Проверка работоспособности

```bash
# Health check
curl http://localhost:5168/health

# Ожидаемый ответ:
# {"status":"healthy","timestamp":"2025-12-09T16:00:00Z","sessions":0}
```

## Тестирование REST API

### Создание сессии

```bash
curl -X POST "http://localhost:5168/api/sessions?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "ru",
    "model": "small",
    "task": "transcribe",
    "use_vad": true
  }'
```

**Ожидаемый ответ:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "websocket_url": "ws://localhost:5168/ws/transcribe/550e8400-e29b-41d4-a716-446655440000?api_key=..."
}
```

### Получение списка сессий

```bash
curl -X GET "http://localhost:5168/api/sessions?api_key=your-secret-api-key-change-in-production"
```

### Получение статуса сессии

```bash
SESSION_ID="550e8400-e29b-41d4-a716-446655440000"
curl -X GET "http://localhost:5168/api/sessions/${SESSION_ID}?api_key=your-secret-api-key-change-in-production"
```

### Получение транскрипта

```bash
curl -X GET "http://localhost:5168/api/sessions/${SESSION_ID}/transcript?api_key=your-secret-api-key-change-in-production"
```

### Закрытие сессии без суммаризации

```bash
curl -X POST "http://localhost:5168/api/sessions/${SESSION_ID}/close?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"generate_summary": false}'
```

### Закрытие сессии с суммаризацией

**Требование:** Ollama должен быть запущен и настроен в `.env`

```bash
curl -X POST "http://localhost:5168/api/sessions/${SESSION_ID}/close?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"generate_summary": true}'
```

### Получение суммаризации

```bash
curl -X GET "http://localhost:5168/api/sessions/${SESSION_ID}/summary?api_key=your-secret-api-key-change-in-production"
```

### Удаление сессии

```bash
curl -X DELETE "http://localhost:5168/api/sessions/${SESSION_ID}?api_key=your-secret-api-key-change-in-production"
```

## Тестирование WebSocket

### Python клиент (пример)

```python
import asyncio
import base64
import json
import wave
import websockets

async def test_websocket():
    # Создайте сессию через REST API сначала
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    api_key = "your-secret-api-key-change-in-production"

    uri = f"ws://localhost:5168/ws/transcribe/{session_id}?api_key={api_key}"

    async with websockets.connect(uri) as websocket:
        # Откройте аудио файл (PCM, 16kHz, mono, int16)
        with wave.open("test_audio.wav", "rb") as wf:
            chunk_size = 1024 * 2  # 2KB chunks

            while True:
                audio_data = wf.readframes(chunk_size)
                if not audio_data:
                    break

                # Отправьте аудио как base64
                message = {
                    "type": "audio_chunk",
                    "audio_data": base64.b64encode(audio_data).decode("utf-8"),
                    "speaker": "Test Speaker"
                }

                await websocket.send(json.dumps(message))

                # Получите транскрипцию
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    print(f"Received: {response}")
                except asyncio.TimeoutError:
                    pass

            # Сообщите об окончании потока
            await websocket.send(json.dumps({"type": "end_of_stream"}))

# Запуск
asyncio.run(test_websocket())
```

## Проверка логов

### Просмотр логов сервера

```bash
# В реальном времени
docker-compose logs -f whisperlive-server

# Последние 100 строк
docker logs whisperlive-server-cpu --tail 100

# Фильтрация по ключевым словам
docker logs whisperlive-server-cpu 2>&1 | grep -E "\[SESSION|TRANSCRIPT|SUMMARY\]"
```

### Ключевые логи для отладки

- `[SESSION_CREATED]` - Сессия создана
- `[WS_CONNECTED]` - WebSocket подключен
- `[WS_DISCONNECTED]` - WebSocket отключен
- `[SESSION_CLOSED]` - Сессия закрыта
- `[TRANSCRIPT]` - Транскрипт сгенерирован
- `[SUMMARY]` - Суммаризация запущена/завершена
- `[ERROR]` - Ошибки

## Проверка ресурсов

### CPU и память

```bash
# Использование ресурсов контейнером
docker stats whisperlive-server-cpu

# Процессы внутри контейнера
docker exec whisperlive-server-cpu ps aux
```

### Дисковое пространство (модели)

```bash
# Размер кэша моделей
du -sh models/ whisper-cache/

# Список скачанных моделей
ls -lh models/
```

## Остановка и очистка

```bash
# Остановка сервера
docker-compose down

# Полная очистка (включая volumes)
docker-compose down -v

# Удаление образа
docker rmi whisperlive-server-cpu
```

## Частые проблемы

### 1. Ошибка "Invalid API key"

**Решение:** Проверьте, что API_KEY в `.env` совпадает с тем, что вы передаете в запросах.

```bash
# Проверить текущий API_KEY
docker exec whisperlive-server-cpu env | grep API_KEY
```

### 2. Ошибка "Session not found"

**Решение:** Сессия могла истечь (TTL = 3600s по умолчанию) или быть удалена.

```bash
# Список активных сессий
curl -X GET "http://localhost:5168/api/sessions?api_key=your-api-key"
```

### 3. Суммаризация не работает

**Проверки:**

```bash
# 1. Ollama запущен?
curl http://localhost:11434/api/tags

# 2. Модель загружена?
ollama list

# 3. Проверить настройки в .env
docker exec whisperlive-server-cpu env | grep OLLAMA
```

### 4. Медленная транскрибация

**Решение:** Для CPU inference используйте модели меньшего размера:

```json
{
  "model": "tiny",  // Самая быстрая
  "model": "base",  // Компромисс
  "model": "small"  // Хорошее качество
}
```

### 5. Out of Memory

**Решение:** Увеличьте лимиты в `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Увеличить с 8G
```

## API документация

FastAPI автоматически генерирует документацию:

- **Swagger UI:** http://localhost:5168/docs
- **ReDoc:** http://localhost:5168/redoc

## Поддержка

При возникновении проблем:

1. Проверьте логи: `docker-compose logs -f`
2. Проверьте здоровье сервера: `curl http://localhost:5168/health`
3. Создайте issue в репозитории с логами и описанием проблемы
