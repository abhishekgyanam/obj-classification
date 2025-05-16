import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.h5")


cap = cv2.VideoCapture(0)


x, y, w, h = 220, 140, 200, 200  

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  

    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (28, 28))
    normalized = resized / 255.0
    input_data = normalized.reshape(1, 28, 28, 1)

    
    prediction = model.predict(input_data, verbose=0)
    digit = np.argmax(prediction)

    
    cv2.putText(frame, f'Prediction: {digit}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
