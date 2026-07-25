import os
import cv2
import numpy as np
import tensorflow as tf

def main():
    model_path = "outputs/road_cleanliness.tflite"
    labels = ["Clean", "Slightly_Dirty", "Very_Dirty"]
    colors = {
        "Clean": (0, 255, 0),         # Green
        "Slightly_Dirty": (0, 165, 255), # Orange
        "Very_Dirty": (0, 0, 255)       # Red
    }

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Make sure to run step 7 (convert_tflite.py) first.")
        return

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Starting webcam... (Press 'q' inside the video window to quit)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess frame for EfficientNet-B0 (224x224, float32, BGR to RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        input_data = np.expand_dims(np.array(resized_frame, dtype=np.float32), axis=0)

        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get best prediction
        best_idx = np.argmax(predictions)
        pred_label = labels[best_idx]
        confidence = predictions[best_idx] * 100.0

        # Draw HUD overlays on the live frame
        h, w, _ = frame.shape
        color = colors.get(pred_label, (255, 255, 255))

        # Top background panel
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

        # Draw classification results
        text = f"{pred_label.upper()}: {confidence:.1f}%"
        cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        # Draw confidence bars for all classes in bottom panel
        cv2.rectangle(frame, (0, h - 80), (w, h), (30, 30, 30), -1)
        for i, label in enumerate(labels):
            y_pos = h - 60 + (i * 22)
            prob = predictions[i]
            bar_width = int(prob * 120)
            
            # Draw label
            cv2.putText(frame, f"{label}:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 222, 220), 1, cv2.LINE_AA)
            
            # Draw progress bar background
            cv2.rectangle(frame, (140, y_pos - 10), (260, y_pos), (70, 70, 70), -1)
            # Draw progress bar fill
            cv2.rectangle(frame, (140, y_pos - 10), (140 + bar_width, y_pos), colors[label], -1)
            # Draw percentage
            cv2.putText(frame, f"{prob * 100.0:.0f}%", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 222, 220), 1, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Road Cleanliness Classifier - Real-Time Test", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()
