# WhisperLive Unified Server

Сервер для транскрибации аудио в реальном времени с поддержкой REST API и WebSocket на едином порту.

## Возможности

- **Единый порт** для REST API и WebSocket соединений
- **Управление сессиями** через REST API
- **Потоковая транскрибация** аудио через WebSocket
- **Определение спикеров** с использованием pyannote
- **GPU ускорение** с поддержкой CUDA
- **Аутентификация** через API ключи
- **Консолидация транскриптов** с удалением дубликатов и группировкой по спикерам
- **Автоматическая суммаризация** встреч с использованием LLM (Ollama/vLLM)


### 2. Настройка API ключа

Отредактируйте `docker-compose.yml` и измените API ключ:

```yaml
environment:
  - API_KEY=your-secret-api-key-change-in-production
```

### 3. Запуск сервера

```bash
docker-compose up -d
```

Сервер будет доступен на порту `5168` (внешний) / `9090` (внутри контейнера).

### 4. Проверка статуса

```bash
curl http://localhost:5168/health
```

## Использование API

### Создание сессии

```bash
curl -X POST "http://localhost:5168/api/sessions?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "ru",
    "model": "large-v3",
    "task": "transcribe",
    "use_vad": true
  }'
```

Ответ:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "websocket_url": "ws://localhost:5168/ws/transcribe/550e8400-e29b-41d4-a716-446655440000?api_key=..."
}
```

### Подключение к WebSocket

После создания сессии подключитесь к WebSocket URL и отправляйте аудио данные:

```json
{
  "type": "audio_chunk",
  "audio_data": "base64_encoded_pcm_audio",
  "speaker": "Спикер1"
}
```

Формат аудио: PCM, 16 kHz, mono, int16

### Получение транскрипта

```bash
curl -X GET "http://localhost:5168/api/sessions/{session_id}/transcript?api_key=your-secret-api-key-change-in-production"
```

Ответ:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "closed",
  "created_at": 1699564800.0,
  "last_activity": 1699564900.0,
  "duration": 100.0,
  "total_segments": 45,
  "transcript": [
    {
      "speaker": "Иван",
      "utterances": [
        {
          "text": "Привет, как дела?"
        }
      ],
      "start": 0.0,
      "end": 5.0
    },
    {
      "speaker": "Мария",
      "utterances": [
        {
          "text": "Хорошо, спасибо. А у тебя?"
        }
      ],
      "start": 5.0,
      "end": 10.0
    },
    {
      "speaker": "Иван",
      "utterances": [
        {
          "text": "Отлично!"
        }
      ],
      "start": 10.0,
      "end": 12.0
    }
  ]
}
```

### Закрытие сессии

Закрытие сессии без суммаризации:

```bash
curl -X POST "http://localhost:5168/api/sessions/{session_id}/close?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"generate_summary": false}'
```

Закрытие сессии с генерацией суммаризации (требует настроенный Ollama/LLM API):

```bash
curl -X POST "http://localhost:5168/api/sessions/{session_id}/close?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"generate_summary": true}'
```

**Важно**: Суммаризация генерируется только если `generate_summary: true` и в сессии есть транскрипт.

Ответ (с суммаризацией):
```json
{
  "session_id": "ae9df18d-8583-4517-8305-69f5488777a8",
  "status": "closed",
  "transcript_available": true,
  "summary": {
    "Название встречи": "Обсуждение проектного роадмапа",
    "Дата и время": "15 октября 2025, 18:03",
    "Участники": ["Иван_Петров", "Мария_Смирнова"],
    "Ключевые темы": [
      "Проектный роадмап",
      "Презентация слайдов"
    ],
    "Принятые решения на встрече и обсуждаемые вопросы": [
      "Иван_Петров - Утвержден план разработки на следующий квартал",
      "Мария_Смирнова - Обсуждали дизайн новой фичи"
    ],
    "Поставленные задачи": [
      "Мария_Смирнова: подготовить финальную версию презентации до 31.12.2025"
    ],
    "model": "qwen2.5:latest",
    "generated_at": "2025-10-15T18:14:01Z"
  }
}
```

### Получение суммаризации

Получить ранее сгенерированную суммаризацию для закрытой сессии:

```bash
curl -X GET "http://localhost:5168/api/sessions/{session_id}/summary?api_key=your-secret-api-key-change-in-production"
```

Ответ:
```json
{
  "session_id": "ae9df18d-8583-4517-8305-69f5488777a8",
  "status": "closed",
  "summary": {
    "Название встречи": "Обсуждение проектного роадмапа",
    "Дата и время": "15 октября 2025, 18:03",
    "Участники": ["Иван_Петров", "Мария_Смирнова"],
    "Ключевые темы": ["Проектный роадмап", "Презентация слайдов"],
    "Принятые решения на встрече и обсуждаемые вопросы": [
      "Иван_Петров - Утвержден план разработки на следующий квартал"
    ],
    "Поставленные задачи": [
      "Мария_Смирнова: подготовить финальную версию презентации до 31.12.2025"
    ],
    "model": "qwen2.5:latest",
    "generated_at": "2025-10-15T18:14:01Z"
  }
}
```

### Получение информации о сессии

```bash
curl -X GET "http://localhost:5168/api/sessions/{session_id}?api_key=your-secret-api-key-change-in-production"
```

### Список всех сессий

```bash
curl -X GET "http://localhost:5168/api/sessions?api_key=your-secret-api-key-change-in-production"
```

### Удаление сессии

```bash
curl -X DELETE "http://localhost:5168/api/sessions/{session_id}?api_key=your-secret-api-key-change-in-production"
```

## Конфигурация

### Параметры запуска сервера

```bash
python3 run_server_unified.py \
  --host 0.0.0.0 \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 10 \
  --max_connection_time 6000 \
  --session_ttl 3600 \
  --cleanup_interval 60
```

Параметры:
- `--host` - хост сервера (по умолчанию: 0.0.0.0)
- `--port` - порт сервера (по умолчанию: 9090)
- `--backend` - бэкенд транскрибации: faster_whisper, tensorrt, openvino
- `--max_clients` - максимальное количество одновременных клиентов
- `--max_connection_time` - максимальное время соединения в секундах
- `--session_ttl` - время жизни сессии в секундах
- `--cleanup_interval` - интервал очистки сессий в секундах
- `--api-keys` - список API ключей (можно указать несколько)


### Настройка Ollama для суммаризации

Для использования функции автоматической суммаризации встреч необходимо настроить Ollama или использовать внешний LLM API.

#### Вариант 1: Локальный Ollama

1. **Установка Ollama**:
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows - скачайте с https://ollama.com/download
```

2. **Загрузка модели**:
```bash
ollama pull qwen2.5:latest
```

3. **Настройка в docker-compose.yml**:
```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=qwen2.5:latest
  # OLLAMA_API_KEY не требуется для локального Ollama
```

#### Вариант 2: Внешний LLM API (например, vLLM)

**Настройка в .env файле**:
```env
OLLAMA_BASE_URL=https://ollama-ui-llm-service.agents.1t.ru/api
OLLAMA_API_KEY=sk-your-api-key
OLLAMA_MODEL=vllm.Qwen3-VL-32B-Instruct-AWQ
```

**Примечание**:
- Сервис использует эндпоинт `/chat/completions` (OpenAI-совместимый формат)
- Имя модели должно совпадать с именем в LLM API (включая префикс `vllm.` если требуется)
- API ключ передается в заголовке `Authorization: Bearer {api_key}`

#### Переменные окружения:

- `OLLAMA_BASE_URL` - URL API для Ollama или совместимого LLM сервиса (по умолчанию: http://localhost:11434)
  - Для локального Ollama: `http://host.docker.internal:11434`
  - Для vLLM API: `https://your-api-url/api` (эндпоинт `/chat/completions` добавляется автоматически)
- `OLLAMA_API_KEY` - API ключ для аутентификации (опционально, для защищенных API)
  - Формат: `Bearer {api_key}` в заголовке Authorization
- `OLLAMA_MODEL` - название модели для суммаризации (по умолчанию: qwen2.5:latest)
  - Примеры: `qwen2.5:latest`, `vllm.Qwen3-80B-A3B-Thinking-int4`

#### Настройка промпта:

Отредактируйте `prompt.yaml` для изменения формата суммаризации и инструкций для LLM.

**Возможности суммаризации**:
- Автоматическое исправление орфографических и пунктуационных ошибок
- Замена неправильно распознанных слов на логичные по контексту
- Улучшение читаемости текста с сохранением исходного смысла
- Выделение ключевых тем, решений и задач
- Определение участников и группировка информации по спикерам

**Примечание**: Если Ollama/LLM API недоступен, функция суммаризации вернет ошибку, но транскрибация будет работать нормально.

## Архитектура

```
┌─────────────────────────────────────────────────┐
│           FastAPI (Единый порт 9090)            │
├─────────────────────────────────────────────────┤
│  REST API Endpoints    │    WebSocket Handler   │
│  - POST /api/sessions  │    - /ws/transcribe/   │
│  - GET /api/sessions   │                        │
│  - GET /transcript     │                        │
│  - POST /close         │                        │
└────────────┬────────────┴────────────┬───────────┘
             │                         │
             └────────┬────────────────┘
                      │
         ┌────────────▼──────────────┐
         │   SessionStore (Менеджер) │
         │   - Управление сессиями   │
         │   - TTL и очистка         │
         │   - Хранение транскриптов │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │ EnhancedTranscriptionServer│
         │   - Whisper Backend       │
         │   - Определение спикеров  │
         │   - Консолидация текста   │
         └───────────────────────────┘
```

## Логирование

Сервер ведет детальное логирование всех событий:

```
2025-01-13 10:30:15 [INFO] rest_api_unified: [SESSION_CREATED] session_id=..., language=ru, model=large-v3
2025-01-13 10:30:16 [INFO] rest_api_unified: [WS_CONNECTED] session_id=..., status=active
2025-01-13 10:32:45 [INFO] rest_api_unified: [SESSION_CLOSED] session_id=..., duration=150.25s, total_segments=87
2025-01-13 10:32:45 [INFO] rest_api_unified: [TRANSCRIPT] session_id=..., total=87, completed=45, blocks=12
```

## Консолидация транскриптов

Система автоматически консолидирует транскрипты:

- **Удаление дубликатов** по временным меткам и тексту
- **Группировка по спикерам** с разделением блоков при смене спикера
- **Разделение по паузам** (пауза > 3 секунд создает новый блок)
- **Ограничение длины** (блок > 30 секунд автоматически разделяется)


