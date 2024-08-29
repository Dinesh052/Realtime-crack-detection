import cv2
import smtplib
import threading
import numpy as np
import scipy.ndimage
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.properties import ObjectProperty
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


# Email configuration
sender_email = "dineshg1125@gmail.com"
sender_password = "zdma pphd gfqr iunv"
receiver_email = "thisispablo1125@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port = 587


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
    data_max = scipy.ndimage.maximum_filter(data, footprint=win, mode='constant')
    data[data != data_max] = 0
    return data


def main_function(gray_image):
    gray_image = gray_image / 255.0
    blur = cv2.GaussianBlur(gray_image, (85, 85), 21)
    gray_image = cv2.subtract(gray_image, blur)

    # Compute Sobel response
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=31)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    # Threshold
    threshold = 4 * 1 * np.mean(mag)
    mag[mag < threshold] = 0

    mag = orientated_non_max_suppression(mag, ang)
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    return result


class RootWidget(TabbedPanel):
    manager = ObjectProperty(None)
    img = ObjectProperty(None)
    img3 = ObjectProperty(None)
    img4 = ObjectProperty(None)
    lab = ObjectProperty(None)

    def on_touch_up(self, touch):
        if not self.img3.collide_point(*touch.pos):
            return True
        else:
            self.lab.text = 'Pos: (%d,%d)' % (touch.x, touch.y)
            return True

    def switch_to(self, header):
        self.manager.current = header.screen

        self.current_tab.state = "normal"
        header.state = 'down'
        self._current_tab = header

    def select_to(self, *args):
        threading.Thread(target=self.process_image, args=(args[1][0],)).start()

    def process_image(self, image_path):
        try:
            print(image_path)
            original_image = cv2.imread(image_path)
            image_gray = cv2.imread(image_path, 0)
            cv2.imwrite(r'E:/EL/crack-detection-beproject-master/GUI/original_im.jpg', original_image)
            result = main_function(image_gray)
            cv2.imwrite(r'E:/EL/crack-detection-beproject-master/GUI/processed_im.jpg', result)

            # Find contours in the processed image
            contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a copy of the original image to draw contours and bounding boxes
            detected_cracks_image = original_image.copy()

            # Draw contours and bounding boxes
            for contour in contours:
                cv2.drawContours(detected_cracks_image, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(detected_cracks_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imwrite(r'E:/EL/crack-detection-beproject-master/GUI/detected_cracks.jpg', detected_cracks_image)

            self.img3.source = r'./processed_im.jpg'
            self.img4.source = r'./detected_cracks.jpg'
            self.img.source = r'./original_im.jpg'

            crack_detected = len(contours) > 0

            if crack_detected:
                self.lab.text = 'Crack Detected: Yes'

                subject = "Crack Detected"
                body = f"A crack has been detected in the image. Metadata:\nImage Path: {image_path}\nNumber of Contours (Cracks): {len(contours)}\n"

                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = receiver_email
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))

                with open(r'E:/EL/crack-detection-beproject-master/GUI/detected_cracks.jpg', 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-Disposition', 'attachment', filename='detected_cracks.jpg')
                    msg.attach(img)

                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
                server.quit()

            else:
                self.lab.text = 'Crack Detected: No'

            count_black = np.sum(result == 0)
            count_white = np.sum(result == 255)
            intensity = count_white * 100 / count_black
            print('Intensity : ', intensity)
            self.img.reload()
            self.img3.reload()
            self.img4.reload()
        except Exception as e:
            print(f"Error: {str(e)}")


class TestApp(App):
    title = 'Feature Extraction'

    def build(self):
        return RootWidget()

    def on_pause(self):
        return True


if __name__ == '__main__':
    TestApp().run()
