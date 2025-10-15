# -*- coding: utf-8 -*-

import os
import time
import random
import concurrent.futures
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import PIL
from PIL.ImageDraw import Draw

# ===================== CONFIG =====================
DATASET_PATH = r"C:\Users\ASUS\Downloads\crowd_project\images.npy"
LABELS_PATH = r"C:\Users\ASUS\Downloads\crowd_project\labels.csv"
FRAMES_DIR = r"C:\Users\ASUS\Downloads\crowd_project\frames" # Update to your local frames directory

# ===================== HELPER FUNCTIONS =====================

def reconstruct_path(image_id: int) -> str:
    """Transforms numerical image ID into full file path."""
    image_id = str(image_id).rjust(6, '0')
    return os.path.join(FRAMES_DIR, f'seq_{image_id}.jpg')

def detect_objects(path: str, model, conf=0.5, iou=0.5):
    """
    Runs YOLOv8 detection on the given image.
    Returns the raw detection results object.
    """
    results = model.predict(source=path, conf=conf, iou=iou, verbose=False)
    return results[0]  # first image result


def count_persons(path: str, model, conf=0.5, iou=0.5) -> int:
    """
    Counts the number of persons in an image using YOLOv8 results.
    Class 0 in COCO = 'person'
    """
    results = model.predict(source=path, conf=conf, iou=iou, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        return 0
    return sum(int(cls) == 0 for cls in boxes.cls)


def filter_detections(data, iou_threshold=0.2, score_threshold=0.7, max_output_size=10):
    """Apply Non-Maximum Suppression (NMS) to remove overlapping boxes."""
    boxes = data['detection_boxes'][0]
    scores = data['detection_scores'][0]
    classes = data['detection_classes'][0]

    selected_indices = tf.image.non_max_suppression(
        boxes, scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    filtered_boxes = tf.gather(boxes, selected_indices)
    filtered_scores = tf.gather(scores, selected_indices)
    filtered_classes = tf.gather(classes, selected_indices)

    return {
        'detection_boxes': filtered_boxes[tf.newaxis, ...],
        'detection_scores': filtered_scores[tf.newaxis, ...],
        'detection_classes': filtered_classes[tf.newaxis, ...],
        'num_detections': [len(selected_indices)]
    }

def get_color(score):
    """Return color based on confidence score."""
    if score > 0.8:
        return 'lime'    # High confidence
    elif score > 0.5:
        return 'yellow'  # Medium confidence
    else:
        return 'red'     # Low confidence

def draw_bboxes(image_path, result, show=True, save_path=None):
    """
    Draws YOLOv8 detections for persons on the image.
    """
    image = PIL.Image.open(image_path).convert("RGB")
    draw = Draw(image)

    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls)
            if cls != 0:  # only 'person'
                continue
            conf = float(box.conf)
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            color = get_color(conf)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            draw.text((x1, y1), f"{conf:.2f}", fill=color)

    if show:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    if save_path:
        image.save(save_path)
    return image


def set_display():
    """Set display options for charts and DataFrames."""
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = 12, 8
    plt.rcParams.update({'font.size': 14})
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.4f}'.format

set_display()

# ===================== DATA LOADING =====================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
images = np.load(DATASET_PATH)
print("Dataset loaded successfully ✅")
print("Type:", type(images))
print("Shape:", images.shape)

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
try:
    data = pd.read_csv(LABELS_PATH)
except UnicodeDecodeError:
    data = pd.read_csv(LABELS_PATH, encoding='latin1')
print("✅ CSV loaded successfully!")
print(data.head())

data['path'] = data['id'].apply(reconstruct_path)
print(data.head())
stats = data.describe()
print(stats)

plt.hist(data['count'], bins=20)
plt.axvline(stats.loc['mean', 'count'], label='Mean value', color='green')
plt.legend()
plt.xlabel('Number of people')
plt.ylabel('Frequency')
plt.title('Target Values')
plt.show()

# ===================== YOLOv8 MODEL LOADING =====================
from ultralytics import YOLO

print("Loading YOLOv8 model...")
detector = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy
print("✅ YOLOv8 model loaded successfully!")

# ===================== EXAMPLES =====================
example_path = data.loc[data['count'] == data['count'].max(), 'path'].iloc[0]
results = detect_objects(example_path, detector)
draw_bboxes(example_path, results)
print(f"Detected persons: {count_persons(example_path, detector)}")


# ===================== MULTIPROCESSING =====================
sample = data.sample(frac=0.1)
start = time.perf_counter()
predictions = []

all_paths = sample['path'].tolist()
results = detector.predict(source=all_paths, conf=0.5, verbose=False)

predictions = []
for res in results:
    boxes = res.boxes
    if boxes is None:
        predictions.append(0)
    else:
        predictions.append(sum(int(cls) == 0 for cls in boxes.cls))


finish = time.perf_counter()
print(f'Finished in {round(finish - start, 2)} second(s).')

sample['prediction'] = predictions
sample['mae'] = (sample['count'] - sample['prediction']).abs()
sample['mse'] = sample['mae'] ** 2

print(f'MAE = {sample['mae'].mean()}\nMSE = {sample['mse'].mean()}')
plt.hist(sample['mae'], bins=20)
plt.title('Absolute Errors')
plt.show()

plt.scatter(sample['count'], sample['prediction'])
plt.xlabel('Actual person count')
plt.ylabel('Predicted person count')
plt.title('Predicted vs. Actual Count')
plt.show()

# ===================== TRAINING PARAMETERS =====================
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 10
LEARNING_RATE = 1e-3
IMAGE_SIZE = 299

def load_image(is_labelled: bool, is_training=True):
    def _get_image(path: str) -> tf.Tensor:
        image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
        if is_training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.3)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.05)
            image = tf.image.resize_with_pad(image, IMAGE_SIZE + 20, IMAGE_SIZE + 20)
            image = tf.image.random_crop(image, size=[IMAGE_SIZE, IMAGE_SIZE, 3])

        return tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    def _get_image_label(img: tf.Tensor, label: int) -> tuple:
        return _get_image(img), label
    return _get_image_label if is_labelled else _get_image

def prepare_dataset(dataset, is_training=True, is_labeled=True):
    image_read_fn = load_image(is_labeled, is_training)
    dataset = dataset.map(image_read_fn, num_parallel_calls=AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(1000)
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

def create_model() -> tf.keras.Model:
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        pooling='avg',
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(512, activation='selu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model

def plot_history(hist):
    mae = hist.history['mean_absolute_error']
    val_mae = hist.history['val_mean_absolute_error']
    x_axis = range(1, len(mae) + 1)
    plt.plot(x_axis, mae, 'bo', label='Training MAE')
    plt.plot(x_axis, val_mae, 'ro', label='Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()
print("✅ TensorFlow training setup ready!")

# ===================== TRAINING =====================
from sklearn.model_selection import train_test_split

# Normalize crowd counts for stability
max_count = data['count'].max()
data['count_norm'] = data['count'] / max_count

# Shuffle + split
data_train, data_valid = train_test_split(
    data, test_size=0.15, random_state=42, shuffle=True
)


ds_train = tf.data.Dataset.from_tensor_slices((data_train['path'], data_train['count_norm']))
ds_valid = tf.data.Dataset.from_tensor_slices((data_valid['path'], data_valid['count_norm']))


ds_train = prepare_dataset(ds_train)
ds_valid = prepare_dataset(ds_valid, is_training=False)

model = create_model()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE,
    restore_best_weights=True)

lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', patience=1, cooldown=1, verbose=1,
    factor=0.75, min_lr=1e-8)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_loss', mode='min'
)


history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduction, tensorboard, checkpoint]
)


mse, mae = model.evaluate(ds_valid)
print(f'Validation MSE (normalized) = {mse}\nValidation MAE (normalized) = {mae}')
print(f'Actual MAE = {mae * max_count:.2f} people')


# ======= Fine-tune last 50 layers =======
base_model = model.layers[0]  # Extract the base CNN
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='mse',
    metrics=['mae']
)

fine_tune_history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=10
)
