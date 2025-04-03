# Segmentacion de ROI 

***Este programa toma nuestra region de interes capturando y reconociendo el objeto dentro de la imagen y procede a segmentarlo de manera individual.
   El segmentador usa el modelo YOLOv11 para detectar y segmentar el objeto, tambien hice uso Optuna para poder mejorar el modelo y de esta forma tener un mejor perfomance aun que por ahora el rendimiento no es el mejor por lo cual sigue en mejora.***

---

![Imagen](https://i.imgur.com/7k5D5Mo.png)


**El modelo es capaz de distinguir distintos objetos dentro de la imagen como es de esperar logrando asi distintas segmentaciones de objetos detectados.**

![Imagen](https://i.imgur.com/RmpLQRs.png)

**Se aplicó un desenfoque Gaussiano antes del Threshold inicial**: *se 
aplica primero un desenfoque Gaussiano a la mascara de probabilidad con valores entre 0 y 1, y luego aplica el threshold. Esto puede crear una transicion mas gradual que resulta en un borde mas suave.*

![Desenfoque](https://i.imgur.com/nT6WuVg.png)

---
**Requerimientos**

| Paquete                 | Versión          | 
| :---------------------- | :--------------- |
| `opencv-contrib-python` | `==4.11.0.86`    |
| `ultralytics`           | `==8.1.0`        |
| `numpy`                 | (Última estable) |
| `optuna`                | (Última estable) |

PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

### Instalación y Configuración

1.  **Clonar el Repositorio (si aplica):**
    ```bash
    git clone https://github.com/m4tuuc/deteccion-interactiva.git
    cd <nombre-del-directorio-del-repositorio>
    ```

2.  **Crear un entorno virtual:**
    ```bash
    # Crear el entorno virtual
    python -m venv venv

    # Activar el entorno virtual
    # En Windows (cmd/powershell):
    .\venv\Scripts\activate
    # En macOS/Linux (bash/zsh):
    source venv/bin/activate
    ```
    
3.  **Instalar Dependencias:**
    Usa el archivo `requirements.txt` proporcionado:
    ```bash
    pip install -r requirements.txt
    ```


---  

3.5 **Creamos el modelo:**
    Dentro del directorio ejecutamos en nuestra terminal: 
    ```
    python optuna.py  
    ```
    
  *Estos nos generara una carpeta runs que contendra nuestro modelo dentro de la carpeta weights/train*

***En mi caso***`D:\runs\segment\train3\weights`
  
---
    
4.  **Preparar Archivos:**
    *   **Imagen de Entrada:** Coloca la imagen que deseas segmentar en el directorio del proyecto (o en una subcarpeta) y asegúrate de que la variable `IMAGE_PATH` dentro del script `segmentacion.py` apunte a ella correctamente.
        ```python
        # Dentro de segmentacion.py
        IMAGE_PATH = 'img.jpg' # <-- Cambia esto si tu imagen se llama diferente o está en otro lugar
        ```
    *   **Modelo YOLOv11:** Coloca tu archivo de modelo entrenado (`.pt`, por ejemplo `best.pt` que generamos con Optuna) en el directorio del proyecto y asegurate de que la variable `MODEL_NAME` apunte a el.
        ```python
        # Dentro de segmentacion.py
        MODEL_NAME = YOLO('./best.pt') # <-- Asegurate que el path sea correcto
        ```

### ▶️ Ejecutar el Programa
**Propósito General:**

*Permitir al usuario seleccionar una Región de Interes (ROI) en la imagen haciendo clic y arrastrando con el boton izquierdo del mouse.
Una vez que el usuario suelta el boton, si el área seleccionada es suficientemente grande, se extrae esa porción de la imagen original para realizar la segmentación sobre ella.*


Una vez que el entorno este activo y las dependencias instaladas, ejecuta el script desde la raíz del directorio del proyecto:

```bash
python segmentacion.py

