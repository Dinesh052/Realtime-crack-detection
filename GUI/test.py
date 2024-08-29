import cv2
import numpy as np
from scipy import ndimage
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import threading
from geopy.geocoders import Nominatim
import requests

# Email configuration
sender_email = "dineshg1125@gmail.com"
sender_password = "zdma pphd gfqr iunv"
receiver_email = "thisispablo@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port = 587

def main_function(gray_image):
    def orientated_non_max_suppression(mag, ang):
        ang_quant = np.round(ang / (np.pi / 4)) % 4
        winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        magE = non_max_suppression(mag, winE)
        magSE = non_max_suppression(mag, winSE)
        magS = non_max_suppression(mag, winS)
        magSW = non_max_suppression(mag, winSW)

        mag[ang_quant == 0] = magE[ang_quant == 0]
        mag[ang_quant == 1] = magSE[ang_quant == 1]
        mag[ang_quant == 2] = magS[ang_quant == 2]
        mag[ang_quant == 3] = magSW[ang_quant == 3]
        return mag

    def non_max_suppression(data, win):
        data_max = ndimage.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max

    gray_image = gray_image / 255.0
    blur = cv2.GaussianBlur(gray_image, (85, 85), 21)
    gray_image = cv2.subtract(gray_image, blur)

    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=31)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    threshold = 8 * np.mean(mag)
    mag[mag < threshold] = 0

    mag = orientated_non_max_suppression(mag, ang)
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    return result

def send_email_with_image(image_path, location):
    subject = "Crack Detected"
    body = f"A crack has been detected in the real-time video feed.\n\nLocation Coordinates:\nLatitude: {location['lat']}\nLongitude: {location['lng']}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as img_file:
        img = MIMEImage(img_file.read())
        img.add_header('Content-Disposition', 'attachment', filename='detected_crack.jpg')
        msg.attach(img)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = main_function(gray_frame)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.1 < aspect_ratio < 10:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if filtered_contours:
        cv2.putText(frame, "Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, True
    else:
        cv2.putText(frame, "No Crack Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, False

def send_email_thread(image_path, location):
    threading.Thread(target=send_email_with_image, args=(image_path, location)).start()

def get_current_location():
    ip_info = requests.get('https://ipinfo.io').json()
    loc = ip_info['loc'].split(',')
    return {'lat': loc[0], 'lng': loc[1]}

def real_time_crack_detection():
    cap = cv2.VideoCapture(0)
    last_email_time = 0
    email_cooldown = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed_frame, crack_detected = process_frame(frame)

        if crack_detected:
            current_time = time.time()
            if current_time - last_email_time > email_cooldown:
                cv2.imwrite('detected_crack.jpg', processed_frame)
                location = get_current_location()
                send_email_thread('detected_crack.jpg', location)
                last_email_time = current_time

        cv2.imshow('Crack Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_crack_detection()
