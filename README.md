# VLA-hack

## Введение

Наша задача — научить манипулятор **SO-101** класть объекты в тарелку. Конкретно: робот должен уметь справляться с тремя заданиями:

| # | Задание |
|---|---------|
| 🧊 | Положить **кубик** в тарелку |
| 🎾 | Положить **теннисный мячик** в тарелку |
| ❓ | Положить **секретный объект** в тарелку *(объект будет раскрыт на финале)* |


Также у нас есть уже собранные датасеты:

- [Google Drive — общий датасет](https://drive.google.com/file/d/17pKcWhjH6h93pfkWoLvjW5kTLeCzqAog/view?usp=sharing)
- [HuggingFace - MuJoCo для кубика, часть 1](https://huggingface.co/datasets/AlexSmirn0v/sim_cube)
- [HuggingFace - MuJoCo для кубика, часть 2](https://huggingface.co/datasets/AlexSmirn0v/sim_cube_v2)
- [HuggingFace - MuJoCo для шарика](https://huggingface.co/datasets/AlexSmirn0v/sym_ball)
- [HuggingFace - реальный датасет для кубика](https://huggingface.co/datasets/AlexSmirn0v/record_1a)
- [HuggingFace - реальный датасет для шарика, часть 1](https://huggingface.co/datasets/AlexSmirn0v/record_1b)
- [HuggingFace - реальный датасет для шарика, часть 2](https://huggingface.co/datasets/AlexSmirn0v/record_2b)
- [HuggingFace - реальный датасет для различных предметов](https://huggingface.co/datasets/AlexSmirn0v/record_1a)

Веса итоговой модели опубликованы на [Google Drive](https://drive.google.com/drive/folders/1rFbzAwFRvQZzDkmboNLFeLGYTcvpp5In?usp=sharing)

Для локальной работы скачайте датасеты в папку `datasets/`:

```bash
mkdir -p datasets
cd datasets
# Вариант 1: через huggingface-cli
huggingface-cli download AlexSmirn0v/YandexLeRobot --repo-type dataset --local-dir .

# Вариант 2: через git (если установлен git-lfs)
git clone https://huggingface.co/datasets/AlexSmirn0v/YandexLeRobot
```

## Старт

### 1. Запись датасета (реальный робот)

Запись демонстраций с использованием leader/follower сетапа:

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=AlexSmirn0v/record_1a \
  --dataset.single_task="Put cube in the plate" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --dataset.num_episodes=40 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
```

**Важные параметры:**
- `--robot.port=/dev/ttyACM1` — порт follower-руки (подставьте свой)
- `--teleop.port=/dev/ttyACM0` — порт leader-руки (подставьте свой)
- `--robot.cameras=...` — две камеры: `front` (`/dev/video6`) и `side` (`/dev/video4`)
- `--dataset.num_episodes=40` — количество записываемых эпизодов
- `--display_data=true` — показывает данные во время записи

### 2. Обучение модели

Обучение SmolVLA на собранном датасете:

```bash
./run_official_smolvla_train_cached.sh \
    --policy.type=smolvla \
    --policy.device=cuda \
    --dataset.repo_id=AlexSmirn0v/record_1a \
    --dataset.root=/app/datasets/real-home \
    --output_dir=/app/outputs/train/smolvla_so101_1 \
    --job_name=smolvla_so101 \
    --batch_size=128 \
    --steps=5000
```

**Важные параметры:**
- `--dataset.repo_id` — ID датасета
- `--output_dir` — куда сохранять чекпоинты
- `--batch_size=128` — размер батча
- `--steps=5000` — количество шагов обучения

Скрипт автоматически:
- Монтирует репозиторий в `/app`
- Пробрасывает GPU
- Сохраняет Hugging Face cache в `outputs/hf_cache`

### 3. Инференс / Оценка модели

Запуск обученной модели для оценки (инференс на реальном роботе):

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --display_data=false \
  --dataset.repo_id=AlexSmirn0v/eval_ya \
  --dataset.single_task="Put object in the plate" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path=/home/user/VLA-hack/001500/pretrained_model
```

**Важные параметры:**
- `--policy.path` — путь к директории с обученной моделью (`pretrained_model`)
- `--display_data=false` — не показывать данные во время инференса

### 4. Оценка в симуляторе (MuJoCo)

Для оценки чекпоинта в симуляции MuJoCo:

```bash
docker run --rm --gpus all \
    -v "$PWD:/app" \
    -w /app \
    dpaleyev/lerobot-workshop:latest \
    python run_smolvla_inference.py \
        --policy-path /app/outputs/train/smolvla_so101_1/checkpoints/last/pretrained_model \
        --dataset-root /app/datasets/real-home \
        --dataset-repo-id AlexSmirn0v/record_1a \
        --episodes 10 \
        --max-steps 250 \
        --fps 10 \
        --summary-path /app/outputs/eval/smolvla_so101_1_summary.json
```

Этот скрипт:
- Загружает обученный SmolVLA checkpoint
- Подтягивает статистики датасета для нормализации
- Прогоняет несколько эпизодов в MuJoCo
- Считает число успешных эпизодов и success rate
- Сохраняет итоговый JSON-отчёт (если передан `--summary-path`)

---

## Сбор датасета в симуляции (MuJoCo)

### Что такое MuJoCo и зачем он здесь

**MuJoCo** (Multi-Joint dynamics with Contact) — физический симулятор для робототехники. Он реалистично считает динамику суставов, контакты и трение, при этом работает очень быстро (десятки тысяч шагов симуляции в секунду) и умеет рендерить изображения с виртуальных камер.

В этом проекте MuJoCo симулирует роборуку **SO-101**: вы управляете ею в виртуальной среде и собираете датасет демонстраций (видео + команды), который потом идёт на обучение VLA-модели.

**Ключевые понятия, которые встретятся в коде:**

| Термин | Что это |
|--------|---------|
| `mjModel` | Статическое описание сцены — тела, суставы, геометрия. Загружается из XML. |
| `mjData` | Текущее состояние симуляции — позиции, скорости, силы. Меняется на каждом шаге. |
| `qpos` | Углы суставов (что сейчас «куда повёрнуто»). |
| `ctrl` | Вектор команд двигателям — то, что вы отправляете руке. |
| `actuator` | Виртуальный двигатель, управляющий суставом. |

Сцена описана в XML-файлах (`asset/`). Каждый объект — отдельный файл, подключаемый через `<include>`. Подробнее — в [MUJOCO.md](./MUJOCO.md).

### Запуск MuJoCo и запись демонстраций

1. Установите зависимости и скачайте ассеты:

```bash
./install.sh
```

2. Активируйте окружение:

```bash
source .venv/bin/activate
```

3. Запустите сбор данных:

```bash
python -m collect_data.run
```

По умолчанию запись идёт в папку `simulation_data_fixed/`, а целевое число эпизодов задаётся в `collect_data/config.py` через `num_demo`.

Что происходит во время записи:

- если `use_master_arm=True`, управление идёт с leader-руки;
- если `use_master_arm=False`, можно управлять с клавиатуры;
- запись эпизода стартует автоматически при первом заметном движении;
- `Z` сбрасывает сцену и очищает текущий несохранённый эпизод;
- `X` принудительно сохраняет текущий эпизод;
- если куб успешно поставлен на тарелку, эпизод сохраняется автоматически.

Клавиши в режиме управления с клавиатуры:

- `Q/A` — `shoulder_pan`
- `W/S` — `shoulder_lift`
- `E/D` — `elbow_flex`
- `I/K` — `wrist_flex`
- `O/L` — `wrist_roll`
- `Space` — открыть/закрыть gripper
- `Z` — reset сцены

### В каком формате сохраняется датасет

Датасет сохраняется в формате **LeRobotDataset**, совместимом с пайплайнами `lerobot`. Корневая папка по умолчанию: `simulation_data_fixed/`.

В каждом кадре записываются:

- `observation.images.front` — фронтальная камера, видео `480x640x3`;
- `observation.images.side` — боковая камера, видео `480x640x3`;
- `observation.state` — вектор `float32` размера `6`;
- `action` — вектор `float32` размера `6`;
- `task` — строка с названием задания, по умолчанию `Put cube on plate`.

Порядок координат в `observation.state` и `action` одинаковый:

- `shoulder_pan.pos`
- `shoulder_lift.pos`
- `elbow_flex.pos`
- `wrist_flex.pos`
- `wrist_roll.pos`
- `gripper.pos`

Единицы измерения:

- первые 5 значений — углы суставов в градусах;
- `gripper.pos` — скаляр в диапазоне `0..100`.

Внутри папки датасета `lerobot` хранит метаданные эпизодов и фич в `meta/`, а изображения кодирует как видеофайлы, а не как отдельные PNG на каждый кадр. Это удобно для последующего обучения и загрузки через `LeRobotDataset.resume(...)`.

### Сбор датасета в реальности

Помимо симуляции, датасет можно собирать на реальном железе. У нас есть такой сетап:

- **Leader-рука** — человек держит её и показывает движение (телеоперация).
- **Follower-рука** — физический робот SO-101, который в реальном времени повторяет движения leader-руки и взаимодействует с объектами.
- **Две камеры** — фиксируют происходящее с разных ракурсов; видео вместе с командами суставов и сохраняется как датасет.

Такой подход называется **имитационным обучением (imitation learning)**: человек один раз показывает правильное поведение, а модель учится его воспроизводить. Для записи и упаковки демонстраций используется библиотека [LeRobot](https://github.com/huggingface/lerobot).

Официальная документация по записи датасета через `lerobot-record`: [LeRobot Record Function](https://huggingface.co/docs/lerobot/il_robots#record-function).

Мы рекомендуем запускать запись так:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, fourcc: MJPG, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, fourcc: MJPG, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader \
    --dataset.repo_id=local/record-test \
    --dataset.root=./record-test \
    --dataset.fps=10 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Put cube on plate" \
    --dataset.push_to_hub=False \
    --display_data=false \
    --dataset.streaming_encoding=false \
    --dataset.vcodec=h264 \
    --dataset.encoder_threads=2 \
    --robot.disable_torque_on_disconnect=false
```

Что здесь важно:

- `--robot.port=/dev/ttyACM1` и `--teleop.port=/dev/ttyACM0` нужно подставить под свои устройства;
- `--robot.cameras=...` задаёт две камеры: `front` и `side`;
- `--dataset.root=./record-test` определяет локальную папку с датасетом;
- `--dataset.repo_id=local/record-test` позволяет работать локально, без отправки на Hub;
- `--dataset.single_task="Put cube on plate"` задаёт текстовое описание задачи;
- `--dataset.streaming_encoding=false` и `--dataset.vcodec=h264` фиксируют предсказуемое локальное кодирование видео.

На Linux, если во время записи датасета не работают клавиши `Left`, `Right` и `Escape`, проверьте, что установлена переменная окружения `$DISPLAY`. Подробнее: [pynput limitations for Linux](https://pynput.readthedocs.io/en/latest/limitations.html#linux).