import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_inference_model(keras_model, img_size=224):
    # This function extracts the base model and head from the trained Keras model
    # and builds a clean inference model without the random augmentation layers.
    print("Building clean inference model (removing augmentation layers for TFLite compatibility)...")
    
    # 1. Create clean inputs
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image_input")
    
    # 2. Find the base model inside the trained Keras model
    # The trained model layers typically are: input -> data_augmentation -> base_model -> pooling -> dense -> dropout -> predictions
    base_model_layer = None
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name or "mobilenet" in layer.name:
            base_model_layer = layer
            break
            
    if base_model_layer is None:
        raise ValueError("Could not locate the base model backbone in the trained Keras model.")
        
    # Re-build base model with clean input shape
    # We load the weights from the trained base model layer
    print(f"Found backbone layer: {base_model_layer.name}")
    
    # Pass inputs directly to the backbone
    x = base_model_layer(inputs)
    
    # Find head layers in the original model
    # We will search for the GlobalAveragePooling2D, Dense, Dropout, and predictions layers
    pooling_layer = None
    fc_layer = None
    predictions_layer = None
    
    for layer in keras_model.layers:
        if isinstance(layer, layers.GlobalAveragePooling2D):
            pooling_layer = layer
        elif isinstance(layer, layers.Dense) and layer.name != "predictions":
            fc_layer = layer
        elif isinstance(layer, layers.Dense) and layer.name == "predictions":
            predictions_layer = layer
            
    if not pooling_layer or not predictions_layer:
        raise ValueError("Could not locate classification head layers in the trained Keras model.")
        
    x = layers.GlobalAveragePooling2D(name=pooling_layer.name)(x)
    if fc_layer:
        x = layers.Dense(fc_layer.units, activation=fc_layer.activation, name=fc_layer.name)(x)
        
    outputs = layers.Dense(predictions_layer.units, activation="softmax", dtype="float32", name="predictions")(x)
    
    inference_model = models.Model(inputs=inputs, outputs=outputs)
    
    # Copy weights layer-by-layer by name
    print("Copying trained weights to inference model...")
    for layer in keras_model.layers:
        try:
            inf_layer = inference_model.get_layer(layer.name)
            inf_layer.set_weights(layer.get_weights())
        except Exception:
            # Skip layers that do not exist in inference model (like input, augmentation, dropout)
            pass
            
    return inference_model

def convert_to_tflite(keras_model_path, tflite_output_path):
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model not found at {keras_model_path}")
        
    print(f"Loading Keras model from {keras_model_path}...")
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Build the clean inference model
    inference_model = build_inference_model(keras_model)
    
    print("Converting Keras model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    # Enable standard optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"TFLite model successfully saved to: {tflite_output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert trained road classifier model to TensorFlow Lite.")
    parser.add_argument("--model", type=str, default="outputs/best_model.keras", help="Path to best classifier model file")
    parser.add_argument("--output", type=str, default="outputs/road_cleanliness.tflite", help="Path to save output TFLite model")
    args = parser.parse_args()
    
    convert_to_tflite(args.model, args.output)

if __name__ == "__main__":
    main()
