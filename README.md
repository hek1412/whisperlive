# WhisperLive Unified Server

Сервер для транскрибации аудио в реальном времени с поддержкой REST API и WebSocket на едином порту.

## Возможности

- **Единый порт** для REST API и WebSocket соединений
- **Управление сессиями** через REST API
- **Потоковая транскрибация** аудио через WebSocket
- **Timeline-based Speaker Tracking** - отслеживание спикеров по временным меткам
- **TensorRT Backend Support** - оптимизированный бэкенд с GPU ускорением
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
  "speaker": "Виталий Александров#29a941be",
  "timestamp": 1765834387.409
}
```

**Формат аудио**: PCM, 16 kHz, mono, int16

**Поля сообщения:**
- `type` - тип сообщения (всегда `"audio_chunk"`)
- `audio_data` - base64-кодированное PCM аудио (int16, 16kHz, mono)
- `speaker` - идентификатор спикера (опционально, по умолчанию "Unknown")
- `timestamp` - клиентская временная метка Unix timestamp (опционально)

**Ответ от сервера:**
```json
{
  "type": "transcription",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "speaker": "Виталий Александров#29a941be",
  "text": "Привет, как дела?",
  "start": "0.000",
  "end": "2.340",
  "completed": false
}
```

**Поля ответа:**
- `type` - тип сообщения (всегда `"transcription"`)
- `session_id` - ID сессии
- `speaker` - спикер для данного сегмента (определяется через speaker timeline)
- `text` - распознанный текст
- `start` - время начала сегмента (в секундах от начала буфера)
- `end` - время окончания сегмента
- `completed` - завершен ли сегмент (false для промежуточных результатов)

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
  --backend tensorrt \
  --trt-model-path ./trt_engines/whisper_large_v3_float16 \
  --session_ttl 3600 \
  --cleanup_interval 60
```

Параметры:
- `--host` - хост сервера (по умолчанию: 0.0.0.0)
- `--port` - порт сервера (по умолчанию: 9090)
- `--backend` - бэкенд транскрибации: faster_whisper, tensorrt, openvino
- `--session_ttl` - время жизни сессии в секундах
- `--cleanup_interval` - интервал очистки сессий в секундах
- `--api-keys` - список API ключей (можно указать несколько)

**TensorRT-специфичные параметры:**
- `--trt-model-path` - путь к директории с TensorRT engine (обязательно для tensorrt backend)
- `--trt-multilingual` - использовать мультиязычную модель (по умолчанию: true)
- `--trt-py-session` - использовать Python session вместо C++ (по умолчанию: false)

**Переменные окружения:**
- `TRT_MODEL_PATH` - путь к TensorRT engine (альтернатива `--trt-model-path`)
- `TRT_MULTILINGUAL` - использовать мультиязычную модель (true/false)
- `TRT_PY_SESSION` - использовать Python session (true/false)


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

## Timeline-based Speaker Tracking (ветка tensorrt-speaker)

### Принцип работы

Вместо традиционного определения спикеров через pyannote, используется **timeline-based подход**:

1. **Клиент отправляет speaker ID** вместе с каждым аудио чанком
2. **Сервер создает speaker timeline** - временную шкалу с метками смены спикера
3. **При форматировании сегментов** сервер определяет спикера по времени начала сегмента

### Структура Speaker Timeline

```python
speaker_timeline = [
    {"offset": 0.0, "speaker": "Виталий Александров#29a941be", "client_ts": 1765834387.409},
    {"offset": 5.23, "speaker": "Организация#42c9aa97", "client_ts": 1765834392.639},
    {"offset": 12.45, "speaker": "Виталий Александров#29a941be", "client_ts": 1765834399.859}
]
```

- `offset` - время в буфере (секунды от начала)
- `speaker` - идентификатор спикера
- `client_ts` - клиентская временная метка (для синхронизации)

### Логика работы

1. **При получении audio_chunk**:
   - Вычисляется `current_buffer_end` (конец текущего буфера)
   - Если спикер изменился, добавляется запись в timeline
   - Аудио добавляется в буфер

2. **При транскрибации**:
   - Whisper обрабатывает аудио и возвращает текст
   - Создается сегмент с временными метками `start` и `end`
   - Вызывается `_get_speaker_at_time(start)` для определения спикера

3. **Очистка timeline**:
   - При клиппировании буфера (> 45 секунд) удаляются старые записи
   - Timeline хранит только актуальные данные

### Преимущества

✅ **Нет зависимости от pyannote** - не требуется GPU модель для speaker diarization
✅ **Точное определение спикера** - клиент точно знает кто говорит
✅ **Низкая задержка** - нет дополнительной обработки на сервере
✅ **Работает с TensorRT** - оптимизированный бэкенд без speaker diarization

### Ограничения

⚠️ **Клиент должен знать спикера** - требуется логика определения спикера на стороне клиента
⚠️ **Чувствительность к галлюцинациям** - если клиент отправляет тишину, Whisper может галлюцинировать
⚠️ **Рекомендуется VAD на клиенте** - фильтровать тишину перед отправкой на сервер

### Рекомендации для клиентов

1. **Используйте VAD (Voice Activity Detection)**:
   ```python
   # Не отправляйте аудио когда нет речи
   if vad_detector.is_speech(audio_chunk):
       send_to_server(audio_chunk, speaker, timestamp)
   ```

2. **Накапливайте чанки**:
   ```python
   # Отправляйте минимум 1.5 секунды аудио
   if buffer_duration >= 1.5:
       send_to_server(buffer, speaker, timestamp)
       buffer.clear()
   ```

3. **Частота отправки**:
   - ❌ Плохо: каждые 20ms (72 чанка/сек) - перегрузка сервера
   - ✅ Хорошо: каждые 1-2 секунды - оптимальная задержка

## Консолидация транскриптов

Система автоматически консолидирует транскрипты:

- **Удаление дубликатов** по временным меткам и тексту
- **Группировка по спикерам** с разделением блоков при смене спикера
- **Разделение по паузам** (пауза > 3 секунд создает новый блок)
- **Ограничение длины** (блок > 30 секунд автоматически разделяется)

## TensorRT Backend (ветка tensorrt-speaker)

### Особенности

- **Минимальная длина чанка**: 1.5 секунды (предотвращает галлюцинации на коротких фрагментах)
- **Нет истории сегментов**: отправляется только текущий сегмент, не последние N
- **Speaker timeline**: автоматическое определение спикера по временным меткам
- **Оптимизация**: TensorRT-LLM для максимальной производительности

### Проблема галлюцинаций

Если клиент отправляет **тишину**, Whisper может галлюцинировать:

```
"Продолжение следует..."
"Субтитры создавал DimaTorzok"
"1, 2, 3, 4, 5..."
```

**Решение**: используйте VAD на клиенте для фильтрации тишины.

### Логирование

```
[WS_RECEIVED] session=..., type=audio_chunk, speaker=Виталий#123, timestamp=..., msg_count=42
[SPEAKER_TIMELINE] Speaker change: Виталий#123 at offset 12.340s (client_ts=1765834387.409)
[WhisperTensorRT:] Processing audio with duration: 1.50s
[WS_SENT] session=..., speaker=Виталий#123, start=12.340, end=13.840, completed=False, text='Привет...'
```


