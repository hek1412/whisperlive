# Инструкция по сборке TensorRT Engine для WhisperLive

## Предварительные требования

- NVIDIA GPU с CUDA support (минимум 8GB VRAM)
- Docker с nvidia-container-toolkit
- ~20GB свободного места на диске

## Шаг 1: Запуск контейнера для сборки

```bash
# Из корня проекта WhisperLive
docker compose -f docker-compose.gpu.yml run --rm --entrypoint /bin/bash whisperlive-server-gpu
```

## Шаг 2: Установка зависимостей (если не в Dockerfile)

```bash
# Установка MPI (нужна для TensorRT-LLM)
apt-get update
apt-get install -y libopenmpi-dev openmpi-bin

# Обновление onnx для совместимости
pip install --upgrade onnx
```

## Шаг 3: Скачивание модели и assets

```bash
cd /app/TensorRT-LLM-examples/examples/whisper

# Создание директории для assets
mkdir -p assets

# Скачивание токенайзера (для multilingual модели)
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken

# Скачивание mel filters
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

# Скачивание модели large-v3 (~3GB, займет 5-10 минут)
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
```

## Шаг 4: Конвертация весов модели

```bash
# Установка переменных окружения
export INFERENCE_PRECISION=float16
export WEIGHT_ONLY_PRECISION=int8  # или float16 для максимального качества
export MAX_BEAM_WIDTH=4
export MAX_BATCH_SIZE=8
export checkpoint_dir=whisper_large_v3_weights_${WEIGHT_ONLY_PRECISION}

# Конвертация PyTorch модели в TensorRT-LLM формат
python3 convert_checkpoint.py \
    --use_weight_only \
    --weight_only_precision $WEIGHT_ONLY_PRECISION \
    --output_dir $checkpoint_dir
```

**Время выполнения:** ~2-5 минут

## Шаг 5: Сборка TensorRT Engines

```bash
# Установка выходной директории
export output_dir=/app/models/whisper_large_v3_${WEIGHT_ONLY_PRECISION}

# Сборка Encoder (займет ~5-10 минут)
trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000

# Сборка Decoder (займет ~5-10 минут)
trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}
```

**Время выполнения:** ~10-20 минут

**Ожидаемые warning:**
```
[TRT] [E] Error Code: 9: Skipping tactic 0x... due to exception...
```
Это нормально - TensorRT просто пропускает несовместимые оптимизации.

## Шаг 6: Копирование assets

```bash
# Копирование токенайзера и фильтров в основную директорию приложения
cp -r assets /app/
```

## Шаг 7: Проверка результата

```bash
# Проверка структуры собранных engines
ls -lh /app/models/whisper_large_v3_${WEIGHT_ONLY_PRECISION}/

# Должны быть директории:
# - encoder/ (с файлами rank0.engine, config.json)
# - decoder/ (с файлами rank0.engine, config.json)
```

## Шаг 8: Выход из контейнера

```bash
exit
```

Engines сохранены в volume `./trt_engines/` на хосте и будут доступны при следующем запуске!

## Шаг 9: Запуск сервера с TensorRT

```bash
# Проверьте что engines на месте
ls -lh ./trt_engines/whisper_large_v3_int8/

# Запустите сервер через docker-compose
docker compose -f docker-compose.gpu.yml up -d

# Проверьте логи
docker logs -f whisperlive-server-gpu
```

## Тестирование

```bash
# Health check
curl http://localhost:5206/health

# Создание сессии
curl -X POST "http://localhost:5206/api/sessions?api_key=your-secret-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"language": "ru", "model": "large-v3", "use_vad": false}'
```

## Сравнение квантизаций

| Квантизация | VRAM | Скорость (RTF) | Качество (WER) | Рекомендация |
|-------------|------|----------------|----------------|--------------|
| **int8** | ~3-4 GB | ~0.05-0.1 | 98-99% от fp16 | ✅ **Рекомендуется для продакшна** |
| **float16** | ~6-8 GB | ~0.1-0.15 | 100% (baseline) | Максимальное качество |
| **int4** | ~2-3 GB | ~0.03-0.05 | 95-97% от fp16 | Экстремальная скорость (экспериментально) |

**RTF** (Real-Time Factor): 0.1 означает, что 10 секунд аудио обрабатываются за 1 секунду.

## Часто задаваемые вопросы

### Q: Что делать если не хватает VRAM?
A: Используйте int8 или уменьшите MAX_BATCH_SIZE до 4 или 2.

### Q: Нужно ли пересобирать при обновлении кода?
A: Нет! Engines собираются один раз. Пересборка нужна только если меняете:
- Модель (например, с large-v3 на medium)
- Квантизацию (int8 → float16)
- MAX_BATCH_SIZE или другие параметры сборки

### Q: Можно ли использовать собранный engine на другом сервере?
A: **НЕТ!** TensorRT engines зависят от:
- Архитектуры GPU (например, RTX 3090 ≠ A100)
- Версии CUDA
- Версии TensorRT

Нужно собирать на той же машине, где будет работать.

### Q: Сколько места занимают engines?
- int8: ~1.5-2 GB
- float16: ~3-4 GB

## Troubleshooting

### Ошибка: `libmpi.so.40: cannot open shared object file`
**Решение:**
```bash
apt-get update && apt-get install -y libopenmpi-dev openmpi-bin
```

### Ошибка: `AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'`
**Решение:**
```bash
pip install --upgrade onnx>=1.15.0
```

### Ошибка: `CUDA out of memory`
**Решение:**
- Закройте другие процессы на GPU
- Уменьшите MAX_BATCH_SIZE до 4 или 2
- Используйте int8 вместо float16

### Engines собраны, но сервер не запускается
**Проверьте:**
1. Путь к engines в docker-compose.gpu.yml совпадает с реальным
2. Assets (multilingual.tiktoken, mel_filters.npz) скопированы в /app/assets/
3. Логи: `docker logs whisperlive-server-gpu`

## Автоматизация

Для автоматической сборки при деплое создайте скрипт или используйте multi-stage Dockerfile с предсобранными engines.

**Внимание:** Сборка в CI/CD требует GPU-enabled runner!




ЕСЛИ ИСПОЛЬЗОВАТЬ build_whisper_tensorrt.sh для сборки В контейнере выполните:
bash build_whisper_tensorrt.sh large-v3 float16 /app/models
Или с другими параметрами:
# Синтаксис:
bash /app/scripts/build_whisper_tensorrt.sh [MODEL_NAME] [QUANTIZATION] [OUTPUT_DIR]

# Примеры:
bash /app/scripts/build_whisper_tensorrt.sh large-v3 float16 /app/models
bash /app/scripts/build_whisper_tensorrt.sh medium int8 /app/models
bash /app/scripts/build_whisper_tensorrt.sh large-v3 int4 /app/models
Скрипт автоматически выполнит все 5 шагов: скачивание assets, скачивание модели, конвертацию, сборку encoder и decoder.



Важные шаги по копированию сборки:
cd /app/TensorRT-LLM-examples/examples/whisper

# Создайте целевую директорию
mkdir -p /app/models/whisper_large_v3_int8

# Скопируйте encoder и decoder
cp -r whisper_large_v3_weights_int8/encoder /app/models/whisper_large_v3_int8/
cp -r whisper_large_v3_weights_int8/decoder /app/models/whisper_large_v3_int8/

# Также нужно скопировать assets (токенайзер и фильтры)
cp -r assets /app/

# Проверьте структуру
ls -lh /app/models/whisper_large_v3_int8/