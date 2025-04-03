import cv2
from ultralytics import YOLO
import numpy as np
import torch
import random

#Cargamos la imagen que queremos segmentar 
IMAGE_PATH = 'mi_imagen.jpg'
#Modelo segmentacion
MODEL_NAME = YOLO('./best.pt')
CONFIDENCE_THRESHOLD = 0.4 #Podemos jugar con esto para encontrar si es que el modelo le cuesta encontrar y reconocer el objeto. 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

selecting_roi = False
roi_start_point = None
roi_end_point = None
final_selected_roi = None
segmentation_results_in_roi = None
roi_origin = None
mask_colors = {}


#Cargamos el modelo
try:
    model = MODEL_NAME
    model.to(DEVICE)
    class_names = model.names
    print(f"Modelo {MODEL_NAME.ckpt_path} cargado en {DEVICE}.")
except Exception as e:
    print(f"Error al cargar modelo YOLOv8: {e}")
    exit()

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"Error al cargar imagen: {IMAGE_PATH}")
    exit()
original_frame = frame.copy()

#Funcion callback raton y ROI
'''
Definimos nuestra funcion encargarda de fomar el ROI basandos en las coordenadas del mouse, tomamos el punto inicial y el final
se las calcula y validamos si es el ancho y alto son mayores a X cantidad de pixeles .

Vamos a buscar nuestra region de interes dentro de la imagen y la vamos a encerrar en un recuadro.

Si la ROI estan en las coordenadas correctas va a retornar la segmentacion del objeto del area que elegimos como nuestro ROI.

'''
def mouse_callback(event, x, y, flags, param):
    global selecting_roi, roi_start_point, roi_end_point, final_selected_roi
    global segmentation_results_in_roi, roi_origin

    current_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
        final_selected_roi = None
        segmentation_results_in_roi = None
        roi_origin = None

    #Actualiza el rectangulo mientras se arrastra
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            roi_end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if selecting_roi:
            selecting_roi = False
            x1 = min(roi_start_point[0], roi_end_point[0])
            y1 = min(roi_start_point[1], roi_end_point[1])
            x2 = max(roi_start_point[0], roi_end_point[0])
            y2 = max(roi_start_point[1], roi_end_point[1])

            if x2 - x1 > 10 and y2 - y1 > 10:
                final_selected_roi = (x1, y1, x2, y2)
                roi_origin = (x1, y1)
                print(f"ROI: {final_selected_roi}. Segmentando...") 

                try:
                    roi_img = original_frame[y1:y2, x1:x2]

                    if roi_img.size == 0:
                        print("Error: ROI vacia.")
                        final_selected_roi = None
                        roi_origin = None
                        return

                    
                    
                    # print("Mostrando ROI enviada. Presiona tecla en esa ventana...")
                    # cv2.waitKey(0)
                    # cv2.destroyWindow("ROI Enviada al Modelo")
                   

                    results = model(roi_img, conf=CONFIDENCE_THRESHOLD, device=DEVICE, verbose=False)

                    if results and results[0].masks is not None:
                         segmentation_results_in_roi = results[0]
                         print(f"Segmentación completada. {len(segmentation_results_in_roi.masks)} mascaras en ROI.")
                    else:
                         print("No se encontraron mascaras en la ROI.")
                         segmentation_results_in_roi = None

                except Exception as e:
                    print(f"Error durante segmentación en ROI: {e}")
                    segmentation_results_in_roi = None
            else:
                print("Selección ROI cancelada (muy pequeña).")
                final_selected_roi = None
                roi_start_point = None
                roi_end_point = None
                roi_origin = None

#Mascara
def get_mask_color(class_id):
    if class_id not in mask_colors:
        mask_colors[class_id] = (random.randint(100, 255), random.randint(100, 255), random.randint(50, 255))
    return mask_colors[class_id]


#VENTANA
window_name = 'Segmentador YOLOv8 Interactivo (Q para salir)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
initial_width = 800
initial_height = 640
cv2.resizeWindow(window_name, initial_width, initial_height)
cv2.setMouseCallback(window_name, mouse_callback)

#Main
while True:
    display_frame = original_frame.copy()
    overlay = display_frame.copy()

    if final_selected_roi:
        cv2.rectangle(display_frame,
                      (final_selected_roi[0], final_selected_roi[1]),
                      (final_selected_roi[2], final_selected_roi[3]),
                      (0, 255, 255), 2)

    if segmentation_results_in_roi and segmentation_results_in_roi.masks and roi_origin:
        masks_data = segmentation_results_in_roi.masks.data.cpu().numpy()
        boxes_data = segmentation_results_in_roi.boxes.data.cpu().numpy()

        h_roi = final_selected_roi[3] - final_selected_roi[1]
        w_roi = final_selected_roi[2] - final_selected_roi[0]

        for i, mask_np in enumerate(masks_data):

            # DEBUG: Visualizar mascara RAW sobre ROI
            # try:
            #     roi_img_debug = original_frame[roi_origin[1]:roi_origin[1]+h_roi, roi_origin[0]:roi_origin[0]+w_roi].copy()
            #     mask_raw_binary = (mask_np > 0.5).astype(np.uint8)
            #     h_roi_dbg, w_roi_dbg = roi_img_debug.shape[:2]
            #     mask_resized_for_dbg = cv2.resize(mask_raw_binary, (w_roi_dbg, h_roi_dbg), interpolation=cv2.INTER_NEAREST)
            #     roi_img_debug[mask_resized_for_dbg == 1] = (0, 0, 255) 
            #     cv2.imshow(f"Mascara {i} sobre ROI Recortada", roi_img_debug)
            #     cv2.waitKey(1)
            # except Exception as e_dbg:
            #     print(f"Error en debug de máscara raw: {e_dbg}")
            # 

            mask_resized = cv2.resize(mask_np, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            if np.sum(mask_binary) == 0:
                continue

            class_id = int(boxes_data[i][5])
            color = get_mask_color(class_id)

            try:
                overlay_roi_slice = overlay[roi_origin[1] : roi_origin[1] + h_roi, roi_origin[0] : roi_origin[0] + w_roi]
                if overlay_roi_slice.shape[0] == h_roi and overlay_roi_slice.shape[1] == w_roi:
                     overlay_roi_slice[mask_binary == 1] = color
                else:
                     print(f"Advertencia: Discrepancia tamaño al dibujar máscara {i}. Slice: {overlay_roi_slice.shape}, Máscara: {(h_roi, w_roi)}")

            except IndexError as e:
                 print(f"Error de indice al dibujar mascara: {e}. roi_origin={roi_origin}, h_roi={h_roi}, w_roi={w_roi}, overlay_shape={overlay.shape}")

        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)


    if selecting_roi and roi_start_point and roi_end_point:
        cv2.rectangle(display_frame, roi_start_point, roi_end_point, (255, 100, 0), 2)

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Programa finalizado.")