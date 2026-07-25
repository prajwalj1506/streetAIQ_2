import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision

def build_data_augmentation(img_size=224):
    # Using Resizing to 256 first, then cropping to img_size (224) for Random Crop effect
    augmentation = models.Sequential([
        layers.Resizing(256, 256),
        layers.RandomCrop(img_size, img_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(15 / 360),
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        layers.RandomContrast(factor=0.15),
        layers.RandomBrightness(factor=0.15),
        layers.GaussianNoise(stddev=0.05)
    ], name="data_augmentation")
    
    return augmentation

def build_classifier(model_name="EfficientNet-B0", num_classes=3, img_size=224, fine_tune_unfreeze=0):
    # Enable mixed precision if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU detected. Enabling mixed precision training...")
        mixed_precision.set_global_policy('mixed_float16')
    else:
        print("No GPU detected. Training using default float32 precision.")
        
    input_shape = (img_size, img_size, 3)
    inputs = layers.Input(shape=input_shape)
    
    # Apply data augmentation
    augmentation = build_data_augmentation(img_size)
    x = augmentation(inputs)
    
    # Select backbone base model
    if model_name == "EfficientNet-B0":
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    elif model_name == "MobileNetV3":
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, input_tensor=x, weights="imagenet")
    elif model_name == "EfficientNetV2B0":
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False, input_tensor=x, weights="imagenet")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    # Initially freeze base model layers
    base_model.trainable = False
    
    # Classifier Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    # Softmax output layer (using float32 activation for stability in mixed precision mode)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model, base_model
