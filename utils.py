import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import wandb, json
import numpy as np
import seaborn as sns
import pandas as pd
import math
import cv2

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from datetime import datetime

from scipy.ndimage import binary_opening, binary_closing, label

TARGET_NAMES = ["background", "foreground"]

def evaluate(model, criterion, data_loader, device):
    """
    Evalúa el modelo en los datos proporcionados y calcula la pérdida promedio.

    Args:
        model (torch.nn.Module): El modelo que se va a evaluar.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        data_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de evaluación.

    Returns:
        float: La pérdida promedio en el conjunto de datos de evaluación.

    """
    model.eval()  # ponemos el modelo en modo de evaluacion
    total_loss = 0  # acumulador de la perdida
    with torch.no_grad():  # deshabilitamos el calculo de gradientes
        for x, y in data_loader:  # iteramos sobre el dataloader
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device)  # movemos los datos al dispositivo
            output = model(x)  # forward pass
            # y_matched = match_mask(output, y) # AJUSTE PARA MASCARA
            # output = match_output_dim(output, y)
            output = F.interpolate(output, size=y.shape[-2:], mode='bilinear', align_corners=False)            
            #total_loss += criterion(output, y.float().unsqueeze(1)).item()  # acumulamos la perdida
            target = y.float()
            
            # 1. Eliminar dimensiones extra de tamaño 1
            if target.ndim > 4:
                target = target.squeeze()
                
            # 2. Asegurar la dimensión del canal
            if target.ndim == 3:
                target = target.unsqueeze(1)
                
            total_loss += criterion(output, target).item()  # acumulamos la perdida

    return total_loss / len(data_loader)  # retornamos la perdida promedio


class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        # if val_loss > self.best_score + delta:
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def print_log(epoch, train_loss, val_loss, current_lr=None):
    print(
        f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Current LR: {current_lr:.5f}"
    )

def match_mask(logits, y):
    # y: (N,H,W) índices
    if y.dim() == 4 and y.size(1) == 1:
        y = y.squeeze(1)
    if logits.shape[-2:] != y.shape[-2:]:
        y = F.interpolate(
            y.unsqueeze(1).float(),  # (N,1,H,W)
            size=logits.shape[-2:],  # (h,w) de la salida
            mode="nearest"
        ).squeeze(1).long()
    return y

def match_target_to_output(output, target):
    """
    Alinea la máscara target al shape del output del modelo para BCEWithLogitsLoss.
    - Convierte a float y normaliza {0,255} -> {0,1}
    - Añade canal si falta
    - Interpola con 'nearest' si el tamaño espacial difiere
    """
    if target.dtype != torch.float32:
        target = target.float()

    # normalizar si vienen como 0/255
    if target.max() > 1:
        target = (target > 0).float()

    # [N,H,W] -> [N,1,H,W]
    if target.ndim == 3:
        target = target.unsqueeze(1)

    # resize si hace falta
    if target.shape[-2:] != output.shape[-2:]:
        target = F.interpolate(target, size=output.shape[-2:], mode="nearest")

    return target

# hacer que la salida adapte do tamano a al tamano del target
def match_output_dim(output, target):
    # print(f"output shape: {output.shape}, target shape: {target.shape}")
    # if len(output.shape) == 4 and len(target.shape) == 3:
    #     # output: [B, C, H, W], target: [B, H, W]
    #     # Reducir canales promediando o tomando el primer canal
    #     if output.shape[1] > 1:
    #         # Promediar los canales para obtener un solo canal
    #         output = output.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    #     else:
    #         # Ya tiene un solo canal, solo mantenerlo
    #         pass
    #     # Ahora output es [B, 1, H, W], necesitamos ajustar dimensiones espaciales
    #     if output.shape[-2:] != target.shape[-2:]:
    #         output = F.interpolate(
    #             output,
    #             size=target.shape[-2:],
    #             mode="bilinear",
    #             align_corners=False
    #         )  # [B, 1, H_target, W_target]
    #     # Eliminar la dimensión del canal para que coincida con target [B, H, W]
    #     output = output.squeeze(1)
    # el
    # if output.shape[-2:] != target.shape[-2:]:
    output = match_output_to_dim(output, target.shape[-2:])
    # print(f"output shape after interpolation: {output.shape}, target shape: {target.shape}")
    return output

def match_output_to_dim(output, dim=800):
    if output.shape[-2:] != dim:
        # Para LOGITS: bilinear.
        output = F.interpolate(output, size=dim, mode="bilinear", align_corners=False)
    return output

def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    scheduler=None,
    do_early_stopping=True,
    patience=5,
    epochs=10,
    log_fn=print_log,
    log_every=1,
    checkpoint_path=None,
    save_optimizer=True,
    loss_ponderada=True
):
    """
    Entrena el modelo utilizando el optimizador y la función de pérdida proporcionados.

    Args:
        model (torch.nn.Module): El modelo que se va a entrenar.
        optimizer (torch.optim.Optimizer): El optimizador que se utilizará para actualizar los pesos del modelo.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        train_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de entrenamiento.
        val_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de validación.
        device (str): El dispositivo donde se ejecutará el entrenamiento.
        patience (int): Número de épocas a esperar después de la última mejora en val_loss antes de detener el entrenamiento (default: 5).
        epochs (int): Número de épocas de entrenamiento (default: 10).
        log_fn (function): Función que se llamará después de cada log_every épocas con los argumentos (epoch, train_loss, val_loss) (default: None).
        log_every (int): Número de épocas entre cada llamada a log_fn (default: 1).
        checkpoint_path: path al archivo donde se guardará el mejor modelo
        save_optmizer: si es true guarda el estado del optimizer en un diccionario

    Returns:
        Tuple[List[float], List[float]]: Una tupla con dos listas, la primera con el error de entrenamiento de cada época y la segunda con el error de validación de cada época.

    """
    try:
        epoch_train_errors = []  # colectamos el error de traing para posterior analisis
        epoch_val_errors = []  # colectamos el error de validacion para posterior analisis
        if do_early_stopping:
            early_stopping = EarlyStopping(
                patience=patience
            )  # instanciamos el early stopping
        best_val_loss = float("inf")  # para trackear el mejor modelo
        
        for epoch in range(epochs):  # loop de entrenamiento
            model.train()  # ponemos el modelo en modo de entrenamiento
            train_loss = 0  # acumulador de la perdida de entrenamiento
            index = 0
            for x, y in train_loader:
                x = x.to(device)  # movemos los datos al dispositivo
                y = y.to(device)  # movemos los datos al dispositivo

                optimizer.zero_grad()  # reseteamos los gradientes

                logits = model(x)                         # [N,1,H,W] logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

                #batch_loss = criterion(logits, y.float().unsqueeze(1))target = y.float()
                target = y.float()
                # 1. Eliminar dimensiones extra de tamaño 1 (ej: [N, 1, 1, H, W] -> [N, H, W])
                if target.ndim > 4:
                    target = target.squeeze()
                    
                # 2. Asegurar la dimensión del canal (ej: [N, H, W] -> [N, 1, H, W])
                if target.ndim == 3:
                    target = target.unsqueeze(1)

                if loss_ponderada:
                    batch_loss = criterion(logits, target, ponderada=True)
                else:
                    batch_loss = criterion(logits, target)
                #batch_loss = criterion(logits, y.float().unsqueeze(1))

                batch_loss.backward()  # backpropagation
                optimizer.step()  # actualizamos los pesos

                train_loss += batch_loss.item()  # acumulamos la perdida
                index += 1

            train_loss /= len(train_loader)  # calculamos la perdida promedio de la epoca
            epoch_train_errors.append(train_loss)  # guardamos la perdida de entrenamiento
            val_loss = evaluate(
                model, criterion, val_loader, device
            )  # evaluamos el modelo en el conjunto de validacion
            epoch_val_errors.append(val_loss)  # guardamos la perdida de validacion
            
            # Guardar mejor modelo
            if checkpoint_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                }
                if save_optimizer:
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

                torch.save(checkpoint, checkpoint_path)
                
                
            if scheduler is not None:
                # si lo configurás para minimizar la loss
                scheduler.step(val_loss)

            if do_early_stopping:
                early_stopping(val_loss)  # llamamos al early stopping

            if log_fn is not None:  # si se pasa una funcion de log
                if epoch == 1 or ((epoch + 1) % log_every == 0):  # loggeamos cada log_every epocas
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_fn(epoch, train_loss, val_loss, current_lr)  # llamamos a la funcion de log

            if do_early_stopping and early_stopping.early_stop:
                print(
                    f"Detener entrenamiento en la época {epoch}, la mejor pérdida fue {early_stopping.best_score:.5f}"
                )
                break

        return epoch_train_errors, epoch_val_errors
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        return None, None


def plot_training(train_errors, val_errors):
    # Graficar los errores
    plt.figure(figsize=(10, 5))  # Define el tamaño de la figura
    plt.plot(train_errors, label="Train Loss")  # Grafica la pérdida de entrenamiento
    plt.plot(val_errors, label="Validation Loss")  # Grafica la pérdida de validación
    plt.title("Training and Validation Loss")  # Título del gráfico
    plt.xlabel("Epochs")  # Etiqueta del eje X
    plt.ylabel("Loss")  # Etiqueta del eje Y
    plt.legend()  # Añade una leyenda
    plt.grid(True)  # Añade una cuadrícula para facilitar la visualización
    plt.show()  # Muestra el gráfico


def model_classification_report(model, dataloader, device, nclasses, output_dict=False, do_confusion_matrix=False):
    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcular precisión (accuracy)
    accuracy = accuracy_score(all_labels, all_preds)
    

    report = classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(nclasses)], 
        output_dict=output_dict
    )
    if not output_dict:
        print(f"Accuracy: {accuracy:.4f}\n")
        print("Reporte de clasificación:\n", report)
    else:
        macroAvg = report["macro avg"]
        return accuracy, macroAvg["precision"], macroAvg["recall"], macroAvg["f1-score"], macroAvg["support"]
        
    # Matriz de confusión
    if do_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        print("Matriz de confusión:\n", cm, "\n")

    return report

def show_tensor_image(tensor, title=None, vmin=None, vmax=None):
    """
    Muestra una imagen representada como un tensor.

    Args:
        tensor (torch.Tensor): Tensor que representa la imagen. Size puede ser (C, H, W).
        title (str, optional): Título de la imagen. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    # Check if the tensor is a grayscale image
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:  # Assume RGB
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(tensors, titles=None, figsize=(15, 5), vmin=None, vmax=None):
    """
    Muestra una lista de imágenes representadas como tensores.

    Args:
        tensors (list): Lista de tensores que representan las imágenes. El tamaño de cada tensor puede ser (C, H, W).
        titles (list, optional): Lista de títulos para las imágenes. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        # Check if the tensor is a grayscale image
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:  # Assume RGB
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()


def plot_sweep_metrics_comparison(accuracies, precisions, recalls, f1_scores, sweep_id, WANDB_PROJECT):
    """
    Crea un gráfico de barras que compara las métricas de rendimiento de diferentes runs de un sweep.
    
    Args:
        accuracies (list): Lista de valores de accuracy para cada run
        precisions (list): Lista de valores de precision para cada run
        recalls (list): Lista de valores de recall para cada run
        f1_scores (list): Lista de valores de f1-score para cada run
        run_names (list): Lista de nombres de los runs
        sweep_id (str): ID del sweep de Weights & Biases
        WANDB_PROJECT (str): Nombre del proyecto de Weights & Biases
    """
   
    
    # Obtener todos los runs del sweep
    api = wandb.Api()
    ENTITY = api.default_entity
    sweep = api.sweep(f"{ENTITY}/{WANDB_PROJECT}/{sweep_id}")

    # Extraer datos de todos los runs
    runs = []
    run_names = []

    for run in sweep.runs:
        if run.state == "finished":  # Solo runs completados
            runs.append(run)
            run_names.append(run.name)

    # Configurar colores para cada métrica
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    metrics = [accuracies, precisions, recalls, f1_scores]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    y_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Crear gráfico combinado
    x = np.arange(len(run_names))  # posiciones de las barras por modelo
    width = 0.2  # ancho de cada barra

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 5))

    # Dibujar cada métrica desplazada
    for i, metric in enumerate(metrics):
        if len(metric) != len(run_names):
            print(f"⚠️ Longitud de {metric_names[i]} ({len(metric)}) no coincide con run_names ({len(run_names)}). Se omite.")
            continue
        ax.bar(x + i*width, metric, width, label=metric_names[i], color=colors[i])

    # Personalización
    ax.set_xlabel("Modelos")
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparación de Métricas por Modelo")
    ax.set_xticks(x + width * (len(metrics)-1)/2)
    ax.set_xticklabels(run_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrar
    plt.tight_layout()
    plt.show()

    # Mostrar información adicional
    print(f"\n=== RESUMEN DE MÉTRICAS ===")
    print(f"Total de runs completados: {len(run_names)}")
    print(f"\n--- Accuracy ---")
    best_accuracy_index = np.argmax(accuracies)
    print(f"Mejor: {run_names[best_accuracy_index]} {accuracies[best_accuracy_index]:.4f}")

    print(f"\n--- Precision ---")
    maxArg = np.argmax(precisions)
    print(f"Mejor: {run_names[maxArg]} {precisions[maxArg]:.4f}")

    print(f"\n--- Recall ---")
    maxArg = np.argmax(recalls)
    print(f"Mejor: {run_names[maxArg]} {recalls[maxArg]:.4f}")

    print(f"\n--- F1-Score ---")
    maxArg = np.argmax(f1_scores)
    print(f"Mejor: {run_names[maxArg]} {f1_scores[maxArg]:.4f}")

    # return best_accuracy_index run id
    print(f"\n\nMejor run ID: {runs[best_accuracy_index].id}")
    return runs[best_accuracy_index].id

def summary_dict(r):
    s = getattr(r, "summary_metrics", None)
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return {}
    if isinstance(s, dict):
        return s
    # fallback para r.summary con wrapper antiguo
    s2 = getattr(getattr(r, "summary", {}), "_json_dict", {})
    if isinstance(s2, dict):
        return s2
    return {}

# define download run function
def download_run(run_id, WANDB_PROJECT, model_name="model.pth"):
    """
    Descarga los pesos de un run de Weights & Biases.
    """
   

    api = wandb.Api()

    ENTITY = api.default_entity  # usá el entity correcto según tu URL

    # 1) Traer el run por path
    run_path = f"{ENTITY}/{WANDB_PROJECT}/{run_id}"
    run = api.run(run_path)

    print("RUN:", run.id, "| name:", run.name)
    print("URL:", run.url)
    print("STATE:", run.state)
    print("CONFIG:", dict(run.config))

    # 2) Leer summary de forma segura (algunas versiones lo devuelven como string)


    summary = summary_dict(run)
    print("SUMMARY KEYS:", [k for k in summary.keys() if not k.startswith("_")])
    print("val_loss:", summary.get("val_loss"))

    # 3) Descargar el modelo de ese run
    #    Si el archivo exacto no existe, listá los .pth disponibles.
    try:
        run.file(model_name).download(replace=True)
        print(f"Descargado: {model_name}")
    except Exception as e:
        print(f"No encontré {model_name} directamente:", e)
        print("Buscando .pth disponibles en el run...")
        pth_files = [f for f in run.files() if f.name.endswith(".pth")]
        for f in pth_files:
            print("->", f.name, f.size)
        if pth_files:
            pth_files[0].download(replace=True)
            print("Descargado:", pth_files[0].name)
        else:
            print("No hay archivos .pth en este run.")

    print("CONFIG:", run.config)
    return run.config


def plot_confusion_matrix(cm, title='Matriz de confusión'):
    """
    Grafica una matriz de confusión.
    """
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,               # mostrar valores
        fmt="d",                  # formato entero
        cmap="RdPu",              # paleta de color
        xticklabels=TARGET_NAMES, # etiquetas en eje X
        yticklabels=TARGET_NAMES  # etiquetas en eje Y
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()




def print_metrics_report(report, title="Reporte de clasificación:"):
    """
    Imprime un DataFrame de métricas (por ejemplo, el classification_report con Dice)
    con formato legible: columnas centradas, espacio adicional, y líneas separadoras.

    Parámetros
    ----------
    report : dict o DataFrame
        Diccionario (como el devuelto por classification_report(output_dict=True))
        o un DataFrame de métricas.
    title : str, opcional
        Título que se muestra antes del reporte (por defecto agrega un emoji 📊).

    Ejemplo
    -------
    print_metrics_report(report_dict)
    """

    # imprimir dice si existe
    if report["macro avg"]["dice"]:
        print(f"Dice: {report['macro avg']['dice']:.4f}\n\n")


    print(title + "\n")

    # Convertir a DataFrame si aún no lo es
    if not isinstance(report, pd.DataFrame):
        df_report = pd.DataFrame(report).T
    else:
        df_report = report.copy()


    # Redondear y ajustar visualmente
    df_report = df_report.round(2)

    # Reemplazar NaN por vacío
    df_report = df_report.replace(np.nan, "", regex=True)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 130,
        "display.colheader_justify", "center",
    ):
        print(df_report.to_string(index=True, justify="center", col_space=12))

    print("=" * 90 + "\n")


def rle_encode(mask):
    pixels = np.array(mask).flatten(order='F')  # Aplanar la máscara en orden Fortran
    pixels = np.concatenate([[0], pixels, [0]])  # Añadir ceros al principio y final
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Encontrar transiciones
    runs[1::2] = runs[1::2] - runs[::2]  # Calcular longitudes
    return ' '.join(str(x) for x in runs)


def predict_and_build_submission(
    model,
    device,
    data_loader,
    out_csv="submission",
    threshold=0.5,
    target_class=1,   # usado solo si el modelo es multiclass
    use_post_proc=False,
    min_size=50,
    debug=False
):
    """
    Genera un submission.csv (con timestamp) a partir de un modelo de segmentación
    que puede ser binario (salida B,1,H,W) o multiclass (salida B,C,H,W).

    - Binario: usa sigmoid y threshold.
    - Multiclass: usa softmax y se queda con `target_class`, luego threshold.

    Args:
        model: modelo de segmentación
        device: torch.device
        img_dir: carpeta con las imágenes de test
        out_csv: nombre base del csv (se le agrega datetime)
        transform: mismas transforms determinísticas que en train (ToTensor, Normalize)
        threshold: umbral para binarizar
        target_class: clase de interés si el modelo tiene C>1 canales
    """
    model.eval()

    image_ids = []
    encoded_pixels = []
    
    debug_pre_masks = []
    debug_post_masks = []
    debug_names = []

    with torch.no_grad():
        for x, name in data_loader:
            x = x.to(device)

            logits = model(x)   # (1,1,256,256)

            # Resize a 800×800 (lo que Kaggle espera)
            H_orig, W_orig = logits.shape[-2], logits.shape[-1]
            logits_big = F.interpolate(
                logits, size=(800, 800), mode="bilinear", align_corners=False
            )

            probs = torch.sigmoid(logits_big)
            mask = (probs > threshold).float()

            # Escalar min_size a la nueva resolución
            # TODO: min_size_scaled no se usa
            if use_post_proc:
                scale_area = (800 * 800) / (H_orig * W_orig)
                min_size_scaled = int(min_size * scale_area)
            else:
                min_size_scaled = min_size            

            #print(f"mask shape: {mask.shape}")
            #print(f"batch shape: {mask.shape[0]} range: {range(mask.shape[0])}")
            # Guardar "pre" para debug
            if debug and use_post_proc:
                # Iterar sobre cada elemento del batch
                for i in range(mask.shape[0]):
                    pre_np = mask[i].squeeze().cpu().numpy().astype(np.uint8)
                    debug_pre_masks.append(pre_np)
                    debug_names.append(name[i])
                
            # ---------- POST-PROCESADO OPCIONAL ----------
            if use_post_proc:
                mask = postprocess_batch(mask, min_size=min_size).to(device)          

            # Guardar "post" para debug
            if debug and use_post_proc:
                # Iterar sobre cada elemento del batch
                for i in range(mask.shape[0]):
                    post_np = mask[i].squeeze().cpu().numpy().astype(np.uint8)
                    debug_post_masks.append(post_np)               

            # Procesar cada elemento del batch
            batch_size = mask.shape[0]
            for i in range(batch_size):
                mask_np = mask[i].squeeze().cpu().numpy().astype(np.uint8)
                rle = rle_encode(mask_np)
                
                image_ids.append(name[i])
                encoded_pixels.append(rle)

    # ==========================
    # PLOTEO EN GRILLA (debug)
    # ==========================
    if debug and use_post_proc and len(debug_pre_masks) > 0:
        n_imgs = len(debug_pre_masks)
        total_slots = 2 * n_imgs  # pre y post
        cols = 6
        rows = math.ceil(total_slots / cols)

        plt.figure(figsize=(cols * 2, rows * 2))  # imágenes más chicas

        for i in range(n_imgs):
            # Índices de subplot (0-based) para pre y post
            pre_idx = 2 * i
            post_idx = 2 * i + 1

            # PRE
            ax_pre = plt.subplot(rows, cols, pre_idx + 1)
            ax_pre.imshow(debug_pre_masks[i], cmap="gray")
            ax_pre.set_title(f"{debug_names[i]} - pre", fontsize=8)
            ax_pre.axis("off")

            # POST
            ax_post = plt.subplot(rows, cols, post_idx + 1)
            ax_post.imshow(debug_post_masks[i], cmap="gray")
            ax_post.set_title(f"{debug_names[i]} - post", fontsize=8)
            ax_post.axis("off")

        plt.tight_layout()
        plt.show()            

    ####### SUBMISSION
    df = pd.DataFrame({"id": image_ids, "encoded_pixels": encoded_pixels})

    # nombre con datetime
    ts = datetime.now().strftime("%d-%m-%Y_%H:%M")
    csv_name = f"submissions/{out_csv}_{ts}.csv"
    df.to_csv(csv_name, index=False)
    print(f"submission guardado como: {csv_name}")

    return df, csv_name


def restaurar_modelo(
    model,
    optimizer=None,
    checkpoint_path="best_model.pt",
    device="cpu",
):
    """
    Carga un modelo (y opcionalmente un optimizador) desde un checkpoint.

    Args:
        model (torch.nn.Module): Instancia del modelo con la misma arquitectura.
        optimizer (torch.optim.Optimizer, opcional): Optimizer a restaurar.
        checkpoint_path (str): Ruta del archivo .pt guardado.
        device (str): Dispositivo donde cargar el modelo.

    Returns:
        model, optimizer, checkpoint: el modelo y optimizador actualizados, y el dict del checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restaurar pesos del modelo
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Restaurar optimizer si corresponde
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint


def continuar_entrenamiento(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    checkpoint_path="best_model.pt",
    do_early_stopping=True,
    patience=5,
    epochs_adicionales=5,
    log_fn=print_log,
    log_every=1,
):
    """
    Restaura el modelo y el optimizer desde un checkpoint y continúa el entrenamiento.

    Args:
        model (torch.nn.Module): Instancia del modelo con la misma arquitectura que el checkpoint.
        optimizer (torch.optim.Optimizer): Optimizer a usar (se sobrescribe con el estado del checkpoint).
        criterion, train_loader, val_loader, device: igual que en train.
        checkpoint_path (str): Ruta del checkpoint a cargar.
        do_early_stopping, patience, epochs_adicionales, log_fn, log_every:
            mismos roles que en `train`, pero para las épocas adicionales.

    Returns:
        (epoch_train_errors, epoch_val_errors, checkpoint_inicial):
            listas de pérdidas de este tramo de entrenamiento
            y el checkpoint desde el que se reanudó.
    """
    # 1) Restaurar estado previo
    model, optimizer, checkpoint = restaurar_modelo(
        model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss_prev = checkpoint.get("val_loss", None)
    print(
        f"Reanudando desde epoch {start_epoch} "
        f"(checkpoint guardado en epoch {checkpoint.get('epoch', 'desconocida')}, "
        f"val_loss={best_val_loss_prev:.5f} )"
    )

    # 2) Continuar entrenamiento por `epochs_adicionales`
    epoch_train_errors, epoch_val_errors = train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        do_early_stopping=do_early_stopping,
        patience=patience,
        epochs=epochs_adicionales,        # solo las épocas nuevas
        log_fn=log_fn,
        log_every=log_every,
        checkpoint_path=checkpoint_path,  # se sigue actualizando el mismo archivo
        save_optimizer=True,
    )

    return epoch_train_errors, epoch_val_errors, checkpoint

def postprocess_batch(preds: torch.Tensor, min_size: int = 50) -> torch.Tensor:
    """
    preds: tensor (B,1,H,W) binario (0/1)
    return: tensor (B,1,H,W) postprocesado
    """
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"postprocess_batch esperaba torch.Tensor, recibió {type(preds)}")

    preds_np = preds.detach().cpu().numpy()  # (B,1,H,W)

    clean_preds = []
    for i in range(preds_np.shape[0]):
        mask_i = preds_np[i, 0]                  # (H,W)
        mask_clean = clean_mask_v2(mask_i, min_size=min_size)
        clean_preds.append(mask_clean[None, ...])  # (1,H,W)

    clean_preds = np.stack(clean_preds, axis=0).astype(np.float32)  # (B,1,H,W)
    clean_preds = torch.from_numpy(clean_preds)                     # tensor (B,1,H,W)
    return clean_preds

def clean_mask(mask_np, min_size=50):
    """
    mask_np: array (H, W) binario {0,1}
    min_size: tamaño mínimo de componente para mantener
    """
    m = mask_np.astype(bool)

    # Opening/closing morfológico
    m = binary_opening(m, structure=np.ones((3, 3)))
    m = binary_closing(m, structure=np.ones((3, 3)))

    # Eliminar componentes muy pequeñas
    labeled, num = label(m)
    for comp in range(1, num + 1):
        if np.sum(labeled == comp) < min_size:
            m[labeled == comp] = False

    return m.astype(np.uint8)

def clean_mask_v2(mask, min_size=5000):
    """
    mask: numpy array binaria 0/1
    """
    # Convert to uint8
    m = mask.astype(np.uint8)

    # ===== 1) Morphological CLOSE (fill small gaps/holes in border)
    #kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)

    # ===== 1) Apertura SUAVE (no fill holes, solo corta ramas finas) =====
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)    

    # ===== 2) Eliminar componentes pequeños =====
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if num_labels > 1:
        cleaned = np.zeros_like(m, dtype=np.uint8)
        for lab in range(1, num_labels):  # 0 es fondo
            if stats[lab, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == lab] = 1
        m = cleaned    

    # ===== 2) Morphological OPEN (remove spurious noise)
    #kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    #m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)

    # ===== 3) Keep LARGEST connected component
    #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    #if num_labels > 1:
    #    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #    m = (labels == largest_idx).astype(np.uint8)

    # ===== 4) Fill small holes
    #m = remove_small_holes(m.astype(bool), area_threshold=4000).astype(np.uint8)

    # ===== 5) Optional final erosion (smooth boundary if too large)
    #kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #m = cv2.erode(m, kernel_erode)

    return m    

def dice_on_val_with_postproc(model, val_loader, device, threshold=0.5, min_size=50):
    model.eval()
    inter = 0.0
    union = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).float()
            if y.ndim == 3:
                y = y.unsqueeze(1)

            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # convertir a numpy para postproceso por imagen
            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()

            # aplicar postproceso por batch
            clean_preds = []
            for i in range(preds_np.shape[0]):
                mask_i = preds_np[i, 0]  # (H,W)
                mask_clean = clean_mask_v2(mask_i, min_size=min_size)
                clean_preds.append(mask_clean[None, ...])  # (1,H,W)

            clean_preds = np.stack(clean_preds, axis=0)   # (B,1,H,W)
            clean_preds = torch.from_numpy(clean_preds).float().to(device)

            inter += (clean_preds * y).sum().item()
            union += (clean_preds.sum() + y.sum()).item()

    dice = (2 * inter + 1e-7) / (union + 1e-7)
    return dice



    
