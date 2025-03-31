import cv2
from ultralytics import YOLO
import numpy as np
import torch

IMAGE_PATH = 'img.jpg'
MODEL_NAME = YOLO('./best.pt')
CONFIDENCE_THRESHOLD = 0.5    


#Tooltip roi
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
TEXT_COLOR = (255, 255, 255) 
BG_COLOR = (50, 50, 50)     
ALPHA = 0.7

selecting_roi = False      
roi_start_point = None      # Punto (x, y) donde empezo la seleccion
roi_end_point = None        # Punto (x, y) actual o final de la seleccion
final_selected_roi = None

hovered_roi_active = False
detections_data = []
current_mouse_pos = (0, 0)
hovered_detection_info = None

#Funcion callback raton
def mouse_callback(event, x, y, flags, param):
    global current_mouse_pos, hovered_detection_info
    global selecting_roi, roi_start_point, roi_end_point, final_selected_roi

    current_mouse_pos = (x, y)

#ROI
def mouse_callback(event, x, y, flags, param):

    global current_mouse_pos, hovered_detection_info, hovered_roi_active
    global selecting_roi, roi_start_point, roi_end_point, final_selected_roi

    current_mouse_pos = (x, y)


    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
        final_selected_roi = None # Borra ROI anterior al empezar
        hovered_detection_info = None 
        hovered_roi_active = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            roi_end_point = (x, y)
            hovered_detection_info = None 
            hovered_roi_active = False
        else:
           
            hovered_detection_info = None
            hovered_roi_active = False    

            for detection in detections_data:
                x1_det, y1_det, x2_det, y2_det = detection['box']
                if x1_det < x < x2_det and y1_det < y < y2_det:
                    hovered_detection_info = detection['info']
                    break 

            #Si no estamos sobre una deteccion, comprobar hover sobre ROI manual
            if hovered_detection_info is None and final_selected_roi:
                x1_roi, y1_roi, x2_roi, y2_roi = final_selected_roi
                if x1_roi < x < x2_roi and y1_roi < y < y2_roi:
                    hovered_roi_active = True # Activar flag de hover sobre ROI

    elif event == cv2.EVENT_LBUTTONUP:
        if selecting_roi:
            selecting_roi = False
            roi_end_point = (x, y)
            x1 = min(roi_start_point[0], roi_end_point[0])
            y1 = min(roi_start_point[1], roi_end_point[1])
            x2 = max(roi_start_point[0], roi_end_point[0])
            y2 = max(roi_start_point[1], roi_end_point[1])

            if x2 - x1 > 5 and y2 - y1 > 5:
                final_selected_roi = (x1, y1, x2, y2)
                print(f"ROI Seleccionado manualmente: {final_selected_roi}")
            else:
                print("Selección de ROI cancelada (área muy pequeña).")
                roi_start_point = None
                roi_end_point = None

#Carga de modelo
try:
    model = MODEL_NAME
    class_names = model.names
    print(f"Modelo {MODEL_NAME} cargado correctamente.")
    print(f"Clases detectables: {list(class_names.values())}")
except Exception as e:
    print(f"Error al cargar el modelo YOLOv8: {e}")
    exit()

#Procesacmiento de la imagen
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"Error: No se pudo cargar la imagen desde {IMAGE_PATH}")
    exit()

print("Ejecutando detección de objetos...")
try:
    # Ejecutar inferencia
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
except Exception as e:
    print(f"Error durante la inferencia de YOLOv8: {e}")
    exit()

# Procesar resultados
for result in results:
    boxes = result.boxes  
    for box in boxes:
        # Coordenadas del cuadro
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]

        conf = float(box.conf[0])

        # Guardar información de la deteccion
        detection_info = {
            'Nombre': class_name,
            'Tipo': class_name, # Puedes diferenciar esto si tienes más info
            'Confianza': f"{conf:.2f}",
            'Info Futura 1': 'Valor Pendiente 1',
            'Info Futura 2': 'Valor Pendiente 2',

        }
        detections_data.append({'box': [x1, y1, x2, y2], 'info': detection_info})

print(f"Detección completada. {len(detections_data)} objetos encontrados.")

#Ventana
window_name = 'Detector YOLOv8 Interactivo (Pulsa Q para salir)'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)




def draw_tooltip(image, text_lines, position):

    line_height = cv2.getTextSize('Tg', FONT, FONT_SCALE, THICKNESS)[0][1] + 5
    max_width = 0
    for line in text_lines:
         (w, h), _ = cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)
         max_width = max(max_width, w)
    tooltip_h = len(text_lines) * line_height + 10
    tooltip_w = max_width + 10

    tooltip_x = position[0] + 15
    tooltip_y = position[1] + 10
    img_h, img_w = image.shape[:2]
    if tooltip_x + tooltip_w > img_w: tooltip_x = position[0] - tooltip_w - 15
    if tooltip_y + tooltip_h > img_h: tooltip_y = position[1] - tooltip_h - 10
    tooltip_x = max(0, tooltip_x)
    tooltip_y = max(0, tooltip_y)

    overlay = image.copy()
    cv2.rectangle(overlay, (tooltip_x, tooltip_y), (tooltip_x + tooltip_w, tooltip_y + tooltip_h), BG_COLOR, -1)
    cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0, image)

    current_line_y = tooltip_y + line_height - 2
    for line in text_lines:
        cv2.putText(image, line, (tooltip_x + 5, current_line_y), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        current_line_y += line_height
    return image # Devuelve la imagen con el tooltip dibujado


while True:
    display_frame = frame.copy()

    #Dibujar cuadros de YOLO
    for detection in detections_data:
        x1, y1, x2, y2 = detection['box']
        color_det = (0, 255, 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_det, 2)

    # Dibuja de ROI
    if selecting_roi and roi_start_point and roi_end_point:
        cv2.rectangle(display_frame, roi_start_point, roi_end_point, (255, 100, 0), 2)
    elif final_selected_roi:
        cv2.rectangle(display_frame, (final_selected_roi[0], final_selected_roi[1]),
                      (final_selected_roi[2], final_selected_roi[3]), (0, 255, 255), 2)

    #Tooltip
    tooltip_drawn = False
    if hovered_detection_info and not selecting_roi:
        info_lines_det = [
            f"Nombre: {hovered_detection_info.get('Nombre', 'N/A')}",
            f"Tipo: {hovered_detection_info.get('Tipo', 'N/A')}",
            f"Confianza: {hovered_detection_info.get('Confianza', 'N/A')}",
            f"Info1: {hovered_detection_info.get('Info Futura 1', 'N/A')}",
            f"Info2: {hovered_detection_info.get('Info Futura 2', 'N/A')}"
        ]
        display_frame = draw_tooltip(display_frame, info_lines_det, current_mouse_pos)
        tooltip_drawn = True


    elif hovered_roi_active and final_selected_roi and not selecting_roi and not tooltip_drawn:
        x1r, y1r, x2r, y2r = final_selected_roi
        wr = x2r - x1r
        hr = y2r - y1r
        info_lines_roi = [
            "ROI Manual",
            f"Nombre: {hovered_detection_info.get('Nombre', 'N/A')}",
            f"Tipo: {hovered_detection_info.get('Tipo', 'N/A')}",
            f"Confianza: {hovered_detection_info.get('Confianza', 'N/A')}"
        ]
        display_frame = draw_tooltip(display_frame, info_lines_roi, current_mouse_pos)
        tooltip_drawn = True



    cv2.imshow(window_name, display_frame)

    # Salir si se presiona 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
print("Ventana cerrada. Programa finalizado.")