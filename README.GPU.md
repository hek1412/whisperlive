# WhisperLive GPU Deployment with TensorRT

Руководство по развёртыванию WhisperLive с TensorRT backend на GPU.

## Требования

### Аппаратные требования
- NVIDIA GPU с поддержкой CUDA 12.1+ (compute capability ≥ 7.0)
- Минимум 8GB GPU VRAM для large-v3 модели
- 32GB RAM
- 50GB свободного места на диске

### Программные требования
- Ubuntu 22.04 / Debian 11+
- Docker 24.0+
- NVIDIA Driver 525+
- NVIDIA Container Toolkit

## Установка

### 1. Установка NVIDIA Driver

```bash
# Проверить текущую версию
nvidia-smi

# Если драйвер не установлен:
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

### 2. Установка Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

### 3. Установка NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 4. Проверка GPU доступности в Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Сборка TensorRT Engine

### Вариант 1: Внутри контейнера (рекомендуется)

```bash
# Запустить контейнер в интерактивном режиме
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm --entrypoint /bin/bash whisperlive-server-gpu

# Внутри контейнера собрать engine
bash /app/build_whisper_tensorrt.sh /app/TensorRT-LLM-examples large-v3 float16

# Скопировать engine на хост (из другого терминала)
docker cp <container_id>:/app/TensorRT-LLM-examples/examples/whisper/whisper_large_v3_float16 ./trt_engines/
```

### Вариант 2: Предсобранный engine

Скачайте предсобранный TensorRT engine и поместите в `./trt_engines/whisper_large_v3_float16/`

## Конфигурация

### Настройка переменных окружения

Создайте `.env`:

```bash
# API Key
API_KEY=your-secure-api-key-here

# LLM для суммаризации
OLLAMA_API_KEY=your-ollama-api-key

# TensorRT настройки
TRT_MODEL_PATH=/app/models/whisper_large_v3_float16
TRT_MULTILINGUAL=true
```

### Настройка docker-compose.gpu.yml

Проверьте пути к TensorRT engine:

```yaml
volumes:
  - ./trt_engines:/app/models  # Здесь должны быть ваши TensorRT engines
```

## Запуск

```bash
# Запустить сервер
docker compose -f docker-compose.gpu.yml up -d

# Проверить логи
docker compose -f docker-compose.gpu.yml logs -f

# Проверить health
curl http://localhost:5169/health
```

## Использование

### Создать сессию

```bash
curl -X POST "http://localhost:5169/api/sessions?api_key=your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "ru",
    "model": "large-v3",
    "use_vad": false
  }'
```

Response:
```json
{
  "session_id": "uuid",
  "websocket_url": "ws://localhost:5169/ws/transcribe/uuid?api_key=..."
}
```

### Подключиться к WebSocket

```javascript
const ws = new WebSocket('ws://localhost:5169/ws/transcribe/uuid?api_key=...');

ws.onopen = () => {
  // Отправлять аудио chunks
  const audioChunk = {
    "type": "audio_chunk",
    "audio": base64_pcm16_data,  // PCM 16kHz mono base64
    "speaker": "User Name"
  };
  ws.send(JSON.stringify(audioChunk));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "backend_ready") {
    console.log("Model loaded, ready to transcribe");
  }

  if (data.type === "transcription") {
    console.log(`${data.speaker}: ${data.text}`);
    console.log(`Completed: ${data.completed}`);
  }
};
```

### Закрыть сессию с суммаризацией

```bash
curl -X POST "http://localhost:5169/api/sessions/uuid/close?api_key=..." \
  -H "Content-Type: application/json" \
  -d '{"generate_summary": true}'
```

## Производительность

### TensorRT vs faster-whisper (large-v3)

| Backend | Device | RTF* | Latency |
|---------|--------|------|---------|
| faster-whisper | CPU (8 cores) | 0.3-0.5 | 3-5s |
| TensorRT (float16) | GPU (RTX 3090) | 0.05-0.1 | 0.5-1s |
| TensorRT (int8) | GPU (RTX 3090) | 0.03-0.07 | 0.3-0.7s |

*RTF (Real-Time Factor): время обработки / длительность аудио. RTF < 1 означает быстрее реального времени.

### Оптимизация

Для лучшей производительности:
1. Используйте **int8 quantization** (2-3x быстрее float16)
2. Отключите VAD если не нужен (`use_vad: false`)
3. Используйте C++ session (`TRT_USE_PYTHON_SESSION=false`)

## Troubleshooting

### GPU не видна в контейнере

```bash
# Проверить nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Out of Memory (OOM)

Уменьшите модель или используйте quantization:
- `large-v3` → `medium` (нужно меньше VRAM)
- `float16` → `int8` или `int4`

### Медленная транскрипция

Проверьте загрузку GPU:
```bash
nvidia-smi -l 1
```

Если GPU использование низкое - возможно узкое место в CPU препроцессинге.

## Мониторинг

### Prometheus метрики (TODO)

```bash
curl http://localhost:5169/metrics
```

### Логи

```bash
docker compose -f docker-compose.gpu.yml logs -f | grep -E "TRANSCRIPTION|WS_SEND|ERROR"
```

## Масштабирование

Для нескольких GPU:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 4
          capabilities: [gpu]
```

Запустите несколько инстансов с разными портами и используйте load balancer (nginx/haproxy).
