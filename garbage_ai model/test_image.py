import os
import argparse
import numpy as np
from PIL import Image

def test_tflite_model(model_path, image_path, labels):
    import tensorflow as tf
    
    # 1. Load TFLite model and allocate tensors
    print(f"Loading TFLite model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 2. Get input and output tensors details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Load and preprocess image
    print(f"Loading and preprocessing image: {image_path}...")
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # EfficientNet expects pixel values in [0, 255] as float32
    input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    # 4. Set tensor input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 5. Run inference
    interpreter.invoke()

    # 6. Retrieve prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    return output_data

def test_keras_model(model_path, image_path, labels):
    import tensorflow as tf
    
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading and preprocessing image: {image_path}...")
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    
    output_data = model.predict(input_data, verbose=0)[0]
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Test trained road cleanliness model on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to test")
    parser.add_argument("--model", type=str, default="outputs/road_cleanliness.tflite", help="Path to TFLite or Keras model file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # Standard labels
    labels = ["Clean", "Slightly_Dirty", "Very_Dirty"]

    # Select runner based on file extension
    try:
        if args.model.endswith('.tflite'):
            predictions = test_tflite_model(args.model, args.image, labels)
        else:
            predictions = test_keras_model(args.model, args.image, labels)
            
        # Display Results
        print("\n==========================================")
        print("                 RESULTS                  ")
        print("==========================================")
        for i, label in enumerate(labels):
            print(f"{label:<15}: {predictions[i] * 100.0:6.2f}%")
            
        best_idx = np.argmax(predictions)
        print("------------------------------------------")
        print(f"Final Decision : {labels[best_idx].upper()} ({predictions[best_idx] * 100.0:.2f}%)")
        print("==========================================\n")
        
    except Exception as e:
        print(f"Error occurred during inference: {e}")

if __name__ == "__main__":
    main()
