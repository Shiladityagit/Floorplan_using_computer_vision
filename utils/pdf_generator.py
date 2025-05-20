import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2

def create_pdf(image, results, class_names):
    pdf_path = "detection_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("YOLOv8 Detection Output")
        plt.axis('off')
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.title("Detection Summary", fontsize=16, pad=20)
        lines = ["Detected Objects:\n"]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                name = class_names[cls_id]
                lines.append(f"{name}: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")

        plt.text(0, 1, "\n".join(lines), fontsize=12, verticalalignment='top')
        pdf.savefig()
        plt.close()

    return pdf_path
