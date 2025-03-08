import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import csv

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return image, gray, edged

def detect_components(image, edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 50 < area < 50000 and 0.2 < aspect_ratio < 5:  # Adjusted filtering
            components.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, components

def detect_connections(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=10, maxLineGap=15)
    return lines

def template_match(image, template_path):
    template = cv2.imread(template_path, 0)
    if template is None:
        print(f"Error: Template image '{template_path}' not found or could not be loaded.")
        return image
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (255, 0, 255), 2)
    return image

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def save_feedback(x, y, label):
    with open("1.0feedback.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([x, y, label])

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        label = input(f"Label component at ({x}, {y}): ")
        save_feedback(x, y, label)
        print(f"Saved feedback: {label} at ({x}, {y})")

def main(image_path, template_paths=[]):
    image, gray, edged = preprocess_image(image_path)
    processed_image, components = detect_components(image, edged)
    text = extract_text(image_path)
    
    # Detect connections (wires)
    lines = detect_connections(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(processed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Match known component templates
    for template in template_paths:
        processed_image = template_match(processed_image, template)
    
    print(f"Detected {len(components)} components.")
    print("Extracted Text:")
    print(text)
    
    cv2.imshow("Processed Circuit Diagram - Click to Label", processed_image)
    cv2.setMouseCallback("Processed Circuit Diagram - Click to Label", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image = "1.0/test_circuit.png"  # Replace with your test circuit image
    template_images = ["resistor_template.png", "capacitor_template.png", "diode_template.png"]  # Replace with actual templates
    main(test_image, template_images)
